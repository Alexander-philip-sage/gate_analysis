import os, datetime, pickle
import numpy as np
import pandas as pd
from globals import RIGHT_AVY_HEADER, LEFT_AVY_HEADER, FREQUENCY, DATA_DIR, all_motion_capture_columns
from globals import COLUMNS_TO_GRAPH, COLUMNS_TO_AREA, COLUMNS_TO_LEG, COLUMNS_BY_SENSOR, MOTION_CAPTURE_COLS
import glob
from scipy.signal import correlate, find_peaks, butter, sosfilt
import random
import matplotlib.pyplot as plt
from time_region import time_region
import time

def extract_trial_data(filename, verbose=False):
  end_part = filename.split('_')[1].replace('.csv','')
  subjectID_str = ''.join([x for x in end_part[:3] if x.isdigit()])
  subjectID =int(subjectID_str)
  end_part= end_part.replace(subjectID_str, '', 1)
  trial =int(''.join([x for x in end_part if x.isdigit()]))
  end_part= end_part.replace(str(trial), '')
  inout = 'outdoors' if ('o' in end_part) else 'indoors'
  pace = end_part[1]
  if pace=='s':
    pace = 'slow'
  elif pace =='f':
    pace ='fast'
  elif pace =='n':
    pace='normal'
  timestamp = datetime.datetime.strptime(filename.split('-')[0], '%Y%m%d')
  if verbose:
    print("filename:", filename)
    print("subjectID:", subjectID)
    print("indoor/outdoor:", inout)
    print("trial:", trial)
    print("pace:", pace)
    print("data collected on:", timestamp)
  return subjectID, inout, pace, trial, timestamp

def check_indoor_and_outdoor(metadata):
  '''removes subjects from the dataset if they don't have at least one indoor
  and one outdoor file'''
  remove_subjectID = []
  for _,subs in metadata.groupby(by=['subjectID']):
    if len(subs[subs.inout=='indoors'])==0 or len(subs[subs.inout=='outdoors'])==0:
      remove_subjectID.append(subs.subjectID.iloc[0])
      print("\n found incomplete data")
      print(subs[['filename', 'subjectID', 'inout', 'pace']])
  return metadata[~metadata.subjectID.isin(remove_subjectID)]

def load_metadata(PACE, DATA_DIR):
  start = time.time()
  ##load all data files
  gate_files = [os.path.basename(x) for x in glob.glob(os.path.join(DATA_DIR,"*.csv"))]
  metadata_list = []
  for file in gate_files:
    subjectID, inout, pace, trial, timestamp = extract_trial_data(file, verbose=False)
    metadata_list.append([file, subjectID, inout, pace, trial, timestamp])
  metadata = pd.DataFrame(metadata_list, columns=['filename', 'subjectID', 'inout', 'pace', 'trial', 'timestamp'])
  print("paces found in raw data", metadata['pace'].unique())
  ##filter on pace
  metadata = metadata.loc[metadata['pace']==PACE]
  ##checks and removes incomplete data
  metadata = check_indoor_and_outdoor(metadata)
  metadata.reset_index(inplace=True, drop=True)
  time_region.track_time("load_metadata", time.time()- start)
  return metadata

def remove_ends_data(df_p,remove_seconds: float =5, verbose=False):
  '''warning, this changes the data in memory every time it is run
  do not run multiple times on the same data
  remove_seconds: seconds on begining and end to remove '''
  if verbose:
    print("shape before removing intro and tail ", df_p.shape)
  n_remove = remove_seconds*FREQUENCY
  df_p.drop(index=df_p.index[:n_remove], axis=0, inplace=True)
  # Using DataFrame.tail() function to drop first n rows
  df_p = df_p.tail(-n_remove)
  if verbose:
    print("shape after removing intro and tail ", df_p.shape)
  return df_p

def low_pass_butterworth(df_b, N: int=4, Wn: float = 20):
  '''pd.options.mode.chained_assignment = None'''
  sos=butter(N, Wn, btype='lowpass', analog=False, output='sos', fs=FREQUENCY)
  dict_df = {}
  for col in df_b.columns:
    if col in COLUMNS_TO_GRAPH:
      filtered = sosfilt(sos, df_b[col].to_numpy().copy())
      dict_df[col] = filtered
    else:
      dict_df[col] = df_b[col].to_numpy()
  return pd.DataFrame(dict_df)

def convert_old_new_naming(new):
  '''converts the new column names to the old ones for compatibility'''
  left_thigh_inx = ".1"
  right_shank_inx = ".2"
  left_shank_inx = ".3"  
  old = ''
  if 'acc' in new:
    old += 'Acceleration '
  elif 'vel' in new:
    old += 'Angular Velocity '
  if new.endswith("_y"):
    old += "Y " 
  elif new.endswith("_z"):
    old += "Z " 
  elif new.endswith("_x"):
    old += "X " 
  if 'acc' in new:
    old += '(m/s^2)'
  elif 'vel' in new:
    old += '(rad/s)'
  
  if new.startswith("Rthigh"):
    pass
  elif new.startswith("Rshank"):
    old += right_shank_inx
  elif new.startswith("Lthigh"):
    old += left_thigh_inx
  elif new.startswith("Lshank"):
    old += left_shank_inx
  if old =='':
    old = new
  return old

def mapping_reorder_columns(df_a: pd.DataFrame):
  reorder_list = []
  current_columns = list(df_a.columns)
  for col_name in COLUMNS_TO_GRAPH:
    assert col_name in current_columns, f"mapping of columns from \n{COLUMNS_TO_GRAPH}\n to \n{current_columns} was done incorrectly"
    reorder_list.append(current_columns.index(col_name))
  
  list_col_indices = list(range(len(current_columns)))
  missing_columns = set(list_col_indices).difference(set(reorder_list))
  reorder_list.extend(list(missing_columns))
  return df_a.iloc[:,reorder_list]

def load_motion_capture_data(filename:str, low_pass: bool = True, N: int=4, Wn:float=20):
  '''takes in only the filename, not the path. this function is modified to accept 
  the motion capture format'''
  start =  time.time()
  df_a = pd.read_csv(os.path.join(DATA_DIR, filename))
  print("columns in raw data file\n", df_a.columns, "\n")
  ##rename cols
  df_a.rename(mapper=convert_old_new_naming, axis='columns', inplace=True)
  # re-order the columns
  df_a = mapping_reorder_columns(df_a)
  print("renamed columns\n", df_a.columns, "\n")
  df_in = df_a
  #df_in = remove_ends_data(df_a)
  if low_pass:
    ret =  low_pass_butterworth(df_in, N=N, Wn=Wn)
    time_region.track_time("load_motion_capture_data", time.time() - start)
    return ret
  else:
    time_region.track_time("load_motion_capture_data", time.time() - start)
    return df_in

def load_data(filename:str, low_pass: bool = True, N: int=4, Wn:float=20):
  '''takes in only the filename, not the path'''
  start =  time.time()
  file_path = os.path.join(DATA_DIR, filename)
  if not os.path.exists(file_path):
    print("expecting to find file at {file_path}")
  df_a = pd.read_csv(file_path, usecols=COLUMNS_TO_GRAPH)
  df_in = remove_ends_data(df_a)
  if low_pass:
    ret =  low_pass_butterworth(df_in, N=N, Wn=Wn)
    time_region.track_time("load_data", time.time() - start)
    return ret
  else:
    time_region.track_time("load_data", time.time() - start)
    return df_in
  
def select_random_df(metadata,data_lookup ):
  subjectID_list = metadata['subjectID'].unique()
  subjectID = subjectID_list[random.randint(0,len(subjectID_list)-1)]
  filenames = metadata[metadata.subjectID==subjectID]['filename'].to_numpy()
  filename = filenames[np.random.randint(low=0, high=(len(filenames)-1))]
  return data_lookup[filename]  , filename

def generate_examples_of_butter_filter(zero_crossing_lookup, metadata,save_dir,  N:int = 4, Wn:float = 20, column:str = LEFT_AVY_HEADER):
  '''  grab a random file and a random gate'''
  for i in range(20):
    random_int = random.randint(0,metadata.shape[0]-1)
    file = metadata['filename'].iloc[random_int]
    subjectID = metadata['subjectID'].iloc[random_int]
    df_filtered = load_data(file, low_pass = True, N=N, Wn=Wn)
    df = load_data(file, low_pass = False)
    zero_crossings = zero_crossing_lookup[file]['right']
    gate_ind = random.randint(0,len(zero_crossings)-2)
    center = zero_crossings[gate_ind][0]
    points = df[column].to_numpy()[center-1*FREQUENCY:center+1*FREQUENCY]
    points_filtered = df_filtered[column].to_numpy()[center-1*FREQUENCY:center+1*FREQUENCY]
    t = np.array([x for x in range(len(points))])/FREQUENCY
    fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,8))
    ax1.plot(t, points)
    ax1.set_title('Raw signal 2s')
    ax1.grid(visible=True)
    ax2.plot(t, points_filtered)
    ax2.set_title(" Butterworth Filter " +" N = "+str(N)+", Wn = "+str(Wn)+"Hz")
    #ax2.set_title(" center " + str(center)+" N="+str(N)+", Wn="+str(Wn))
    ax2.grid(visible=True)
    print()
    _=fig.suptitle("Subject " +str(subjectID)+" " + column+ " " + COLUMNS_TO_AREA[column])
    filter_dir = os.path.join(save_dir, "butterworth_examples")
    if not os.path.exists(filter_dir):
      os.mkdir(filter_dir)
    image_name = file.replace(".csv", '')+ " center " +str(center)+'.png'
    fig.savefig(os.path.join(filter_dir, image_name) )

if __name__=="__main__":
  load_motion_capture_data("20201115_CP01_noVSSRstimulus_006.csv")
  #print(all_motion_capture_columns)

  #print([convert_old_new_naming(x) for x in all_motion_capture_columns])