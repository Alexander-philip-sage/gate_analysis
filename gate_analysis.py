import os
import glob
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import Counter
import random
from typing import List, Tuple
from scipy.spatial import distance
#from scipy.signal import resample
from scipy.signal import correlate, find_peaks, butter, sosfilt
from scipy.interpolate import interp1d
from scipy.stats import shapiro, ttest_rel, wilcoxon
import statsmodels.api as sm
from collections import defaultdict
import pickle
from poincare import poincare


RIGHT_AVY_HEADER = 'Angular Velocity Y (rad/s).2'
LEFT_AVY_HEADER = 'Angular Velocity Y (rad/s).3'
COLUMNS_TO_GRAPH = ['Acceleration Y (m/s^2)',  'Angular Velocity Y (rad/s)',##right thigh 
                    'Acceleration Y (m/s^2).1','Angular Velocity Y (rad/s).1',##left thigh
                    'Acceleration Y (m/s^2).2', RIGHT_AVY_HEADER,##right shank
                    'Acceleration Y (m/s^2).3',LEFT_AVY_HEADER ]##left shank
tmp = []
for val in COLUMNS_TO_GRAPH:
  tmp.append(val.replace('Y','X'))
  tmp.append(val.replace('Y','Z'))
COLUMNS_TO_GRAPH.extend(tmp)

COLUMNS_TO_AREA = {  'Acceleration Y (m/s^2)':'right thigh',  'Angular Velocity Y (rad/s)':'right thigh',##right thigh 
                    'Acceleration Y (m/s^2).1':'left thigh','Angular Velocity Y (rad/s).1':'left thigh',##left thigh
                    'Acceleration Y (m/s^2).2':'right shank', RIGHT_AVY_HEADER:'right shank',##right shank
                    'Acceleration Y (m/s^2).3':'left shank',LEFT_AVY_HEADER:'left shank' }##left shank     
tmp = list(COLUMNS_TO_AREA.items())
for val in tmp:
  COLUMNS_TO_AREA[val[0].replace('Y','X')] = val[1]
  COLUMNS_TO_AREA[val[0].replace('Y','Z')] = val[1]
COLUMNS_TO_LEG = {}
for k in list(COLUMNS_TO_AREA.keys()):
  COLUMNS_TO_LEG[k]=COLUMNS_TO_AREA[k].split(' ')[0]
COLUMNS_BY_SENSOR = [{'sensor':'thigh','right':'Acceleration Y (m/s^2)','left': 'Acceleration Y (m/s^2).1'}, 
                     {'sensor':'thigh','right':'Angular Velocity Y (rad/s)', 'left':'Angular Velocity Y (rad/s).1'},
                     {'sensor':'shank','right':'Acceleration Y (m/s^2).2','left':'Acceleration Y (m/s^2).3'},
                     {'sensor':'shank','right':RIGHT_AVY_HEADER, 'left':LEFT_AVY_HEADER},]    
tmp = COLUMNS_BY_SENSOR.copy()        
for d in tmp:
   COLUMNS_BY_SENSOR.append({'sensor':d['sensor'], 'right':d['right'].replace('Y','X'), 'left':d['left'].replace('Y','X')})
   COLUMNS_BY_SENSOR.append({'sensor':d['sensor'], 'right':d['right'].replace('Y','Z'), 'left':d['left'].replace('Y','Z')})   

DATA_DIR = os.path.join(r"C:\Users\sage\Documents\gate_analysis\raw_data")
GATE_CROSSING = -0.3
FREQUENCY = 128


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
  found_subjects = set(metadata['subjectID'].unique())
  diff_subjects = set([i for i in range(1,31)]).difference(found_subjects)
  if len(diff_subjects)>0:
    print("expected 30 subjects. these were missing", diff_subjects)
  for _,subs in metadata.groupby(by=['subjectID']):
    if len(subs[subs.inout=='indoors'])==0 or len(subs[subs.inout=='outdoors'])==0:
      remove_subjectID.append(subs.subjectID.iloc[0])
      print("\n found incomplete data")
      print(subs[['filename', 'subjectID', 'inout', 'pace']])
  return metadata[~metadata.subjectID.isin(remove_subjectID)]


def load_metadata(PACE, DATA_DIR):
  ##load all data files
  gate_files = [os.path.basename(x) for x in glob.glob(os.path.join(DATA_DIR,"*.csv"))]
  assert len(gate_files)>0
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

def load_data(filename:str, low_pass: bool = True, N: int=4, Wn:float=20):
  '''takes in only the filename, not the path'''
  df_a = pd.read_csv(os.path.join(DATA_DIR, filename), usecols=COLUMNS_TO_GRAPH)
  df_in = remove_ends_data(df_a)
  if low_pass:
    return low_pass_butterworth(df_in, N=N, Wn=Wn)
  else:
     return df_in
  
def select_random_df(metadata,data_lookup ):
  subjectID_list = metadata['subjectID'].unique()
  subjectID = subjectID_list[random.randint(0,len(subjectID_list)-1)] 
  filenames = metadata[metadata.subjectID==subjectID]['filename'].to_numpy()
  filename = filenames[np.random.randint(low=0, high=(len(filenames)-1))]
  return data_lookup[filename]  , filename

def generate_examples_of_butter_filter(zero_crossing_lookup, metadata,  N:int = 4, Wn:float = 20, column:str = LEFT_AVY_HEADER):
  ##grab a random file and a random gate
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
    filter_dir = os.path.join(SAVE_DIR, "butterworth_examples")
    if not os.path.exists(filter_dir):
      os.mkdir(filter_dir)
    image_name = file.replace(".csv", '')+ " center " +str(center)+'.png'
    fig.savefig(os.path.join(filter_dir, image_name) )

########################################################################################
########################################################################################


def find_swing_stance_index(signal, min_percent=35):
  '''finds the peak right before the valley of the angular velocity y 
  data stream.
  input: signal from a single gate
  returns: index and value'''
  min_index = int(min_percent*len(signal)/100)
  max_index, valley_value = find_lowest_valley(signal)
  peak_indices = find_peaks(signal)[0]
  possible_stance_change_index = []
  edge_start = 3
  edge_end = len(signal)-edge_start -1
  for x in peak_indices:
    if (edge_end > x > edge_start) and (max_index > x > min_index):
      possible_stance_change_index.append(x)
  peak_values = [signal[x] for x in possible_stance_change_index]
  if len(peak_values)==0:
    #plt.plot(signal)
    #pname = "signal_np.pickle"
    #with open(pname, 'wb') as fileobj:
    #  pickle.dump(np.array(signal),fileobj )
    raise ValueError("couldn't find peak.min index {} max index {} peak indices ".format(min_index,max_index)+str(peak_indices))

  stance_change_index =  possible_stance_change_index[peak_values.index(max(peak_values))]
  stance_change_value = signal[stance_change_index]
  return stance_change_index, stance_change_value

def find_lowest_valley(signal):
  '''finds the min peak in a signal.
  returns the x,y values from that signal where the min peak is'''
  assert (type(signal)==list or type(signal)==np.ndarray)
  inv_signal = -1*signal
  peak_indices = find_peaks(inv_signal)[0]
  edge_start = 1
  edge_end = len(signal)-edge_start -1
  peak_indices = [x for x in peak_indices if (edge_start < x < edge_end)]
  peak_values = [signal[x] for x in peak_indices]
  min_index =peak_values.index(min(peak_values))
  return peak_indices[min_index],peak_values[min_index]

def max_peak(signal, edge_start = 3):
  '''finds the max peak in a signal.
  returns the x,y values from that signal where the max peak is'''
  peak_indices = find_peaks(signal)[0]
  edge_end = len(signal)-edge_start -1
  peak_indices = [x for x in peak_indices if (edge_start < x < edge_end)]
  peak_values = [signal[x] for x in peak_indices]
  max_index =peak_values.index(max(peak_values))
  return peak_indices[max_index],peak_values[max_index]


def extract_gate_crossings(df: pd.core.frame.DataFrame, header:str, gate_crossing: float = -0.6) -> List[int]:
  '''input
  gate_crossing: set the zero crossing below zero to avoid false positives
  returns: zero_crossings: a list of all the indices when a value in df crosses from below gate_crossing to above it
  '''
  zero_crossings = []
  for i, x in enumerate(df[header].to_list()):
    if i < df.shape[0]-1:
      #before = df[header].iloc[i]
      #after = df[header].iloc[i+1]
      ##zero crossing detected
      if df[header].iloc[i-2:i].mean() < gate_crossing and df[header].iloc[i:i+2].mean() > gate_crossing:
        zero_crossings.append(i)
  return zero_crossings

def pair_gate_ends(zero_crossings: List[int], sensor: str = None, filename: str=None, gate_length_bounds: dict =None ) -> List[Tuple[int]]:
  if gate_length_bounds:
    min_gate_length = gate_length_bounds[filename][sensor][0]
    max_gate_length = gate_length_bounds[filename][sensor][1]
  zero_pairs = []
  save= True
  for i in range(len(zero_crossings)-1):
    if not gate_length_bounds:
      save = True
    else:
      gate_length = zero_crossings[i+1]- zero_crossings[i]
      if min_gate_length < gate_length < max_gate_length:
        save = True
      else:
        save = False
    if save:
      zero_pairs.append((zero_crossings[i], zero_crossings[i+1]))
  return zero_pairs

def check_shape_zero_crossings(zero_gd, dstream):
  '''checks that there is a valley at all, and that there exists a 
  peak to the left of the valley and the right of ~30% of the gate data points'''
  passed_zeros = []
  for zero_pair in zero_gd:
    gate_d =   dstream[zero_pair[0]:zero_pair[1]]
    try:
      find_swing_stance_index(gate_d)
      v_i, v_v = find_lowest_valley(gate_d)
      assert v_i > int(0.5*len(gate_d)), "stance valley index must be greatest and on the right"
    except ValueError:
      pass
    except AssertionError:
      pass
    else:
      passed_zeros.append(zero_pair)
  return passed_zeros


def calc_all_gate_crossings(metadata, data_lookup: dict, gate_crossing: float = GATE_CROSSING, gate_length_bounds: dict =None):
  '''detects gate crossings for each leg for every file. 
  creates a lookup dict that allows for looking up gate values given a filename and leg
  saves zero crossings as tuples, not as a list, this way the tuples can be 
  filtered based on min,max gate lengths
  '''
  zero_crossing_lookup = {}
  for filename in metadata['filename'].to_list():
    df_1 = data_lookup[filename]
    ##zero crossings are lists here
    zero_crossings_right = extract_gate_crossings(df_1,RIGHT_AVY_HEADER, gate_crossing= gate_crossing)
    zero_crossings_left = extract_gate_crossings(df_1,LEFT_AVY_HEADER, gate_crossing= gate_crossing)
    ##conver zero crossings to tuples
    zero_crossings_right = pair_gate_ends(zero_crossings_right, sensor = RIGHT_AVY_HEADER,filename = filename, gate_length_bounds=gate_length_bounds)
    zero_crossings_left = pair_gate_ends(zero_crossings_left, sensor =LEFT_AVY_HEADER, filename = filename, gate_length_bounds=gate_length_bounds)
    ##check that the gates have the right shape
    zero_crossings_right = check_shape_zero_crossings(zero_crossings_right, df_1[RIGHT_AVY_HEADER].to_numpy())
    zero_crossings_left = check_shape_zero_crossings(zero_crossings_left, df_1[LEFT_AVY_HEADER].to_numpy())

    zero_crossing_lookup[filename]={'right':zero_crossings_right, 'left':zero_crossings_left}
  return zero_crossing_lookup


def graph_zero_crossings(zero_crossings: List[Tuple[int]], avy_data,filename:str, save_dir: str = 'gate_crossings', window: int = 50, graph_limit: int = 20, gate_max: int = None):
  '''graph_limit : number of graphs to make
  gate_max: if set, sets a max gate size. this allows you to create graphs of only small gates to look for sources of errors
  returns: None. saves and displays graphs
  '''
  max_gates_display = 3
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  assert(len(zero_crossings)>max_gates_display),"not enough zero crossings to plot"
  for i in range(3,len(zero_crossings)):
    if len(zero_crossings)>4:
      start = zero_crossings[i][0]
      end = zero_crossings[i][1]
      if (not gate_max) or ((end-start )< gate_max):
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(25,7))
        zero_cross_indices = [avy_data.index[zero_crossings[i][0]], avy_data.index[zero_crossings[i][1]]]
        zero_cross_values = [avy_data.iloc[zero_crossings[i][0]],avy_data.iloc[zero_crossings[i][1]] ]        
        ax[0].plot(avy_data.iloc[end-window:end+window])
        ax[0].set_title(label='around recent zero')
        ax[0].scatter([zero_cross_indices[1]],[zero_cross_values[1]]  , color='red')
        ax[1].plot(avy_data.iloc[start:end])
        ax[1].set_title(label='prev 1 cycle')   
        ax[1].scatter(zero_cross_indices,zero_cross_values  , color='red')
        ax[2].plot(avy_data.iloc[zero_crossings[i-max_gates_display][1]:end])
        ax[2].scatter(zero_cross_indices,zero_cross_values  , color='red')
        ax[2].set_title(label='prev 3 cycles')
        ax[2].scatter(zero_cross_indices,zero_cross_values  , color='red') 
        fig.suptitle("graph " + "{:,}".format(graph_limit) + ". "+filename+" inspecting row "+"{:,}".format(end) + ". gate size "+str(end-start))
        fig.savefig(os.path.join(save_dir,str(graph_limit)+'.png'))
        graph_limit -=1
        if graph_limit==0:
          break


def avg_std_gate_lengths(zero_crossings: List[Tuple[int]]):
  points_p_gate = [x[1]-x[0] for x in zero_crossings]
  points_p_gate = np.array(points_p_gate)
  return  points_p_gate.mean() , points_p_gate.std(), points_p_gate.max(), points_p_gate.min()  


def hist_gate_lengths(zero_crossing: List[Tuple[int]]):
  avg, std , m, n = avg_std_gate_lengths(zero_crossing)
  print("mean:", round(avg,2), "std:",round(std,2), " max:", m, "min:", n )
  points_p_gate = [x[1]-x[0] for x in zero_crossing]
  plt.hist(points_p_gate)
  _ = plt.title("Histogram of Gate Size")
  plt.xlabel("size of gate (data points)")
  plt.ylabel("# of gates detected")

def find_gate_crossing_threshold(data_lookup, metadata: pd.core.frame.DataFrame, save_dir_gate_lengths:str):
  '''to find out how the gate crossing threshold impacts the stats of the gate
     lengths, calculate the average gate lengths with varying gate_crossing values'''
  stats_data = []
  for gate_crossing in [0,-0.3, -0.6, -0.9, -1.2]:
    zero_crossing_lookup=calc_all_gate_crossings(metadata, data_lookup, gate_crossing = gate_crossing)
    for sensor_cols in [RIGHT_AVY_HEADER, LEFT_AVY_HEADER]:
      for i, inout in enumerate(['indoors', 'outdoors']):
        all_gate_lengths = []
        for x in metadata[metadata['inout']==inout].groupby(by=['subjectID', 'pace']):
          df_trials=x[1]
          for filename in df_trials['filename'].to_numpy():
            zero_crossings = zero_crossing_lookup[filename][COLUMNS_TO_LEG[sensor_cols]]
            points_p_gate = [x[1]-x[0] for x in zero_crossings]
            all_gate_lengths.extend(points_p_gate)
        all_gate_lengths = np.array(all_gate_lengths)
        stats_data.append([sensor_cols, COLUMNS_TO_AREA[sensor_cols], inout, all_gate_lengths.mean(), all_gate_lengths.std(), all_gate_lengths.max(), all_gate_lengths.min(), gate_crossing])
  stats_df = pd.DataFrame(stats_data, columns=['sensor_name', 'area', 'inout', 'avg gate length (data points)', 'std', 'max', 'min', 'vertical_gate_crossing'])
  stats_df.to_csv(os.path.join(save_dir_gate_lengths, "gate_lengths_per_vertical_crossing.csv"))
  return stats_df

def graphing_gate_crossing_thresholds(stats_df: pd.core.frame.DataFrame, save_dir_gate_lengths: str):
  for column_to_graph in [RIGHT_AVY_HEADER, LEFT_AVY_HEADER]:
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(15,5))
    for i, inout in enumerate(['indoors', 'outdoors']):
      wherec = ((stats_df['inout']==inout) & (stats_df['sensor_name']==column_to_graph))
      avg = stats_df[wherec]['avg gate length (data points)']
      std =  stats_df[wherec]['std']
      mx =  stats_df[wherec]['max']
      x_gates =stats_df[wherec]['vertical_gate_crossing']
      #fig, ax = plt.subplots()
      ax[i].plot(x_gates ,avg, color='black')
      ax[i].fill_between(x_gates, avg-std, avg+std, alpha=0.4, color='gray')
      ax[i].set_xlabel("gate crossing threshold")
      ax[i].set_ylabel("gate size (data points)")
      title = column_to_graph+ ' '+COLUMNS_TO_AREA[column_to_graph]
      _= ax[i].set_title( title+ ' ' + inout)     
      fig.savefig(os.path.join(save_dir_gate_lengths,title.replace("/", '-')+'.png'))

def stats_gate_lengths_by_file(metadata,data_lookup,  df_cols: List[str], save_dir_gate_lengths: str, fname_gate_length_file: str = "per_file", MAX_STD: float = 2, zero_crossing_lookup: dict =None):
  '''for each of the two main sensors, create a csv with the gate length 
    stats for each file, also track the thresholds for each file of what
    defines an outlier'''
  filter_to_gate_thresh = {}
  for filename in metadata['filename']:
    filter_to_gate_thresh[filename] = {}
  print("gate crossing used", GATE_CROSSING)
  if not zero_crossing_lookup:
    zero_crossing_lookup=calc_all_gate_crossings(metadata,data_lookup, gate_crossing = GATE_CROSSING)
  for sensor_cols in [RIGHT_AVY_HEADER, LEFT_AVY_HEADER]:
    df_per_file = pd.DataFrame([], columns =df_cols )
    per_filename =fname_gate_length_file+'_'+ sensor_cols.replace("/", '-')+".csv"  
    for i, inout in enumerate(['indoors', 'outdoors']):
      for x in metadata[metadata['inout']==inout].groupby(by=['subjectID', 'pace']):
        df_trials=x[1]
        for filename in df_trials['filename'].values:
          subjectID, inout, pace, trial, timestamp =extract_trial_data(filename)
          zero_crossings = zero_crossing_lookup[filename][COLUMNS_TO_LEG[sensor_cols]]
          avg, std , ppgmax, ppgmin = avg_std_gate_lengths(zero_crossings)
          row = [sensor_cols,COLUMNS_TO_AREA[sensor_cols], inout, filename, trial, subjectID, avg,std,ppgmax, ppgmin,data_lookup[filename].shape[0], GATE_CROSSING]
          filter_to_gate_thresh[filename][sensor_cols] = (avg-MAX_STD*std, avg+MAX_STD*std)
          df_per_file = pd.concat([df_per_file, pd.DataFrame([row], columns=df_cols)])
    df_per_file.sort_values('std',inplace=True, ignore_index=True )
    save_fname = os.path.join(save_dir_gate_lengths,per_filename)
    print("saving to", save_fname)
    df_per_file.to_csv(save_fname, index=False)
  print("saving file", fname_gate_length_file, " with gate length stats per file")
  return df_per_file, filter_to_gate_thresh


def plot_gate_lengths_p_subject(filename,save_dir_gate_lengths):
  df_gate_lengths = pd.read_csv(os.path.join(save_dir_gate_lengths,filename))
  print("stats of gate lengths (data Points)")
  print(df_gate_lengths['avg gate length (data points)'].describe())
  print("\nstd of gate lengths (data Points)")
  print(df_gate_lengths['std'].describe())
  df_gate_lengths['std/avg'] =df_gate_lengths.apply(lambda x: x['std']/x['avg gate length (data points)'], axis=1) 
  print("\nstats of std/avg")
  percentiles = df_gate_lengths['std/avg'].describe()
  print(percentiles)
  fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(15,5))
  ax[0].scatter(df_gate_lengths['subjectID'].values,df_gate_lengths['std/avg'].values )
  _ = ax[0].set_xlabel("SubjectID")
  _ = ax[0].set_ylabel("std/avg")
  _ = ax[0].set_title("Variation in gate lengths per subject: "+df_gate_lengths['area'].iloc[0]+ " "+df_gate_lengths['sensor'].iloc[0])
  outliers_df = df_gate_lengths[df_gate_lengths['std/avg']>percentiles['75%']].copy()
  print("number of outliers defined by 75 percentile:", outliers_df.shape[0])
  print("outliers by indoors/outdoors")
  ct_indoors = outliers_df[outliers_df['in-out']=='indoors'].shape[0]
  print("\tindoors", ct_indoors, "{}%".format(round(100*ct_indoors/outliers_df.shape[0]),2))
  print("\toutdoors", outliers_df.shape[0]-ct_indoors, "{}%".format(round(100*(outliers_df.shape[0]-ct_indoors)/outliers_df.shape[0]),2))
  outlier_p_subject = Counter(outliers_df['subjectID'].values)
  osub = list(outlier_p_subject.keys())
  oct = [outlier_p_subject[x] for x in osub]
  ax[1].bar(osub,oct )
  _ = ax[1].set_xlabel("SubjectID")
  _ = ax[1].set_ylabel("count outlier files")
  _ = ax[1].set_title("75 Percentile outliers")
  fig.savefig(os.path.join(save_dir_gate_lengths,"std-avg_outlier_graph_"+df_gate_lengths['area'].iloc[0]+".png"))
  return df_gate_lengths
########################################################################################
########################################################################################

def find_missing(condition, name, metadata, PACE):
  one_to_thirty = set([x for x in range(1,31)])
  subjects_found = set(metadata[condition]['subjectID'].unique())
  subjects_missing = one_to_thirty.difference(subjects_found)
  print(PACE)
  print(name)
  print("subjects missing", subjects_missing)

def count_data_points_saved(metadata, data_lookup, zero_crossing_lookup):
  columns = ['filename', 'subjectID', 'inout', 'pace', 'trial', 'sensor', 'raw_data_points', 'good_data_points', 'percent_good']
  data = []
  for i, row in metadata.iterrows():
    
    for sensor in [LEFT_AVY_HEADER, RIGHT_AVY_HEADER]:
      new_row = {}
      #print("sensor", sensor)
      filename = row.filename
      t=data_lookup[filename][sensor]
      t=t.dropna()
      raw_data_points = len(t)
      zero_crossings=zero_crossing_lookup[filename][COLUMNS_TO_LEG[sensor]]
      data_points_saved = sum([tup[1]-tup[0] for tup in zero_crossings])
      new_row['raw_data_points'] = raw_data_points
      new_row['good_data_points'] = data_points_saved
      new_row['percent_good']=round(data_points_saved*100/raw_data_points,2)
      new_row['sensor']=sensor
      new_row['filename']=filename
      new_row['subjectID']= row['subjectID']
      new_row['inout']=row['inout']
      new_row['pace']=row['pace']
      new_row['trial']=row['trial']
      data.append(new_row)
  df_good_data_ct = pd.DataFrame(data)
  return df_good_data_ct

def describe_sensors(good_points_df):
  sensors = list(good_points_df['sensor'].unique())
  for sensor in sensors:
    sub_df = good_points_df[good_points_df['sensor']==sensor]
    print(sensor)
    print("avg", round(sub_df['percent_good'].mean(),2))
    print("std", round(sub_df['percent_good'].std(),2))
    print("max", sub_df['percent_good'].max())    
    print("min", sub_df['percent_good'].min())

########################################################################################
########################################################################################

def grab_normalized_gate(df: pd.core.frame.DataFrame, zero_crossings: List[Tuple[int]], gate_ind:int , header:str):
  '''grabs the gate at the index given and normalizes the data points to 100'''
  start = zero_crossings[gate_ind][0]
  end   = zero_crossings[gate_ind][1]
  raw_series = df[header].iloc[start:end]
  raw_values = raw_series.values
  raw_index = list(raw_series.index)
  interp = interp1d(raw_index, raw_values.copy())
  sampled = interp(np.linspace(min(raw_index), max(raw_index),num=100))
  return raw_values, sampled

def calc_avg_std_gates(df: pd.core.frame.DataFrame, zero_crossings: List[int] , header:str):
  '''aggregates all the sampled gates together for one column/sensor after
  normalizing the length of the gate to 100 data points'''
  ct_gates = len(zero_crossings)-1
  all_gates = np.zeros((ct_gates,100))
  for i in range(ct_gates):
    _, all_gates[i] = grab_normalized_gate(df,zero_crossings, i, header)
  avg = all_gates.mean(axis=0)
  std = all_gates.std(axis=0)
  return avg, std , all_gates

def graph_normalizing_affect(zero_crossing_lookup,data_lookup):
  ##graph a gate at random
  ##graph the raw signal and the signal normalized to 100 points via upsampling or downsampling

  ##tried using resample, but it was more prone to introducing noise and didn't work with upsampling
  ##interpolation works with both upsampling and downsampling
  df  , filename = select_random_df(metadata, data_lookup)
  zero_crossings_right= zero_crossing_lookup[filename][COLUMNS_TO_LEG[RIGHT_AVY_HEADER]]
  zero_crossings_left= zero_crossing_lookup[filename][COLUMNS_TO_LEG[LEFT_AVY_HEADER]]
  print("number gates - right leg", len(zero_crossings_right)-1)
  print("number gates - left leg", len(zero_crossings_left)-1)
  gate_ind = random.randint(0,len(zero_crossings_right)-2)
  raw_values, sampled = grab_normalized_gate(df,zero_crossings_right, gate_ind, RIGHT_AVY_HEADER)
  fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(15,5))
  ax[0].plot(raw_values)
  ax[0].set_title("raw signal")
  ax[1].plot(sampled)
  ax[1].set_title("down sampled signal")
  _=fig.suptitle("gate "+str(gate_ind)+ ' '+RIGHT_AVY_HEADER+ ' '+COLUMNS_TO_AREA[RIGHT_AVY_HEADER])

def label_axis(column_to_graph, ax):
  if 'Velocity' in column_to_graph:
    ax.set_ylabel("rad/s")
  elif 'Acceleration' in column_to_graph:
    ax.set_ylabel("m/s^2")
  else:
    print(column_to_graph)
  ax.set_xlabel('normalized time')

def grab_peak_valley(signal):
  peak_index,peak_value = find_swing_stance_index(signal)
  valley_index,valley_value = find_lowest_valley(signal)
  pv_i = [peak_index, valley_index]
  pv_v = [peak_value, valley_value]
  return pv_i, pv_v

def plot_avy_peak_singlefile(subjectID: int, metadata, data_lookup, zero_crossing_lookup):
  '''plot the avg across all gates for the primary data streams 
  on the left and right foot'''
  df , filename= select_random_df(metadata,data_lookup )
  zero_crossings_left=zero_crossing_lookup[filename]['left']
  zero_crossings_right= zero_crossing_lookup[filename]['right']
  avg_left, std_left, all_gates_left  = calc_avg_std_gates(df, zero_crossings_left,LEFT_AVY_HEADER)
  avg_right, std_right, all_gates_right  = calc_avg_std_gates(df, zero_crossings_right, RIGHT_AVY_HEADER)
  print("shape of all gates right", all_gates_right.shape)
  print("shape of all gates left", all_gates_left.shape)
  print("plotted in red are the max peak and min valley")
  fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(15,5))
  ax[0].plot(avg_left, color='black')
  ax[0].fill_between(np.arange(0,100), avg_left-std_left, avg_left+std_left, alpha=0.4, color='gray')
  ax[0].scatter(*grab_peak_valley(avg_left), color='red', s=100, alpha=1)
  ax[0].set_title(LEFT_AVY_HEADER+' '+COLUMNS_TO_AREA[LEFT_AVY_HEADER])
  label_axis(LEFT_AVY_HEADER, ax[0])
  ax[1].plot(avg_right, color='black')
  ax[1].fill_between(np.arange(0,100), avg_right-std_right, avg_right+std_right, alpha=0.4, color='gray')
  ax[1].scatter(*grab_peak_valley(avg_right), color='red', s=100, alpha=1)
  ax[1].set_title(RIGHT_AVY_HEADER+ ' '+COLUMNS_TO_AREA[RIGHT_AVY_HEADER])
  label_axis(RIGHT_AVY_HEADER, ax[1])
  _ = fig.suptitle("average gate plots for file "+file)


##exploring other data streams with the gates defined above
def graph_random_thighs(metadata,data_lookup):
  df , filename= select_random_df(metadata,data_lookup )
  zero_crossings_left=zero_crossing_lookup[filename]['left']
  zero_crossings_right= zero_crossing_lookup[filename]['right']
  avg_right, std_right, all_gates_right = calc_avg_std_gates(df, zero_crossings_right,COLUMNS_TO_GRAPH[0])
  avg_left, std_left, all_gates_left  = calc_avg_std_gates(df, zero_crossings_left,COLUMNS_TO_GRAPH[2])
  fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(15,5))
  ax[0].plot(avg_left, color='black')
  ax[0].fill_between(np.arange(0,100), avg_left-std_left, avg_left+std_left, alpha=0.4, color='gray')
  ax[0].set_title(COLUMNS_TO_GRAPH[2]+ ' '+COLUMNS_TO_AREA[COLUMNS_TO_GRAPH[2] ])
  label_axis(COLUMNS_TO_GRAPH[2], ax[0])
  ax[1].plot(avg_right, color='black')
  ax[1].fill_between(np.arange(0,100), avg_right-std_right, avg_right+std_right, alpha=0.4, color='gray')
  ax[1].set_title(COLUMNS_TO_GRAPH[0]  +' '+COLUMNS_TO_AREA[COLUMNS_TO_GRAPH[0] ])
  label_axis(COLUMNS_TO_GRAPH[0], ax[1])

  _ = fig.suptitle("average gate plots for file "+filename)


def aggregate_single_subject(data_lookup, metadata: pd.core.frame.DataFrame, zero_crossing_lookup: dict, sensor:str, inout:str , subjectID: int):
  '''given one subject, pace, and inout. it combines all the data across all the files
  inout could be either [indoors, outdoors]
  to combine different trials, filter on 'subjectID', 'pace', 'inout', 
  if more than one row exists, concatenate the data from the rows'''
  pace = metadata['pace']
  assert len(metadata['pace'].unique())==1, "paces"+str(metadata['pace'].unique())
  where_cond = ((metadata['subjectID']==subjectID)&(metadata['inout']==inout))
  records = metadata[where_cond]
  #print('analyzing file(s) ', records.values)
  if records.shape[0]==1:
    filename =records['filename'].iloc[0]
    df_1 = data_lookup[filename]
    zero_crossings = zero_crossing_lookup[filename][COLUMNS_TO_LEG[sensor]]
    avg, std , all_gates = calc_avg_std_gates(df_1, zero_crossings, sensor)
  elif records.shape[0]==2:
    df_1 = data_lookup[ records['filename'].iloc[0]]
    zero_crossings_1 = zero_crossing_lookup[records['filename'].iloc[0]][COLUMNS_TO_LEG[sensor]]
    _, _ , all_gates_1 = calc_avg_std_gates(df_1, zero_crossings_1, sensor)
    df_2 = data_lookup[records['filename'].iloc[1]]
    zero_crossings_2 =zero_crossing_lookup[records['filename'].iloc[1]][COLUMNS_TO_LEG[sensor]]
    _, _ , all_gates_2 = calc_avg_std_gates(df_2, zero_crossings_2, sensor)   
    all_gates = np.concatenate((all_gates_1,all_gates_2), axis=0 )
    #avg = all_gates.mean(axis=0)
    #std = all_gates.std(axis=0)     
  elif records.shape[0]==0:
    errstr = "\nno data found\nsubject {} inout {} pace {}".format(subjectID, inout, pace)
    raise Exception("unexpected number of files "+ str(records)+errstr)   
  else:
    errstr = "\nsubject {} inout {} pace {}".format(subjectID, inout, pace)
    raise Exception("unexpected number of files "+ str(records)+errstr)   

  return all_gates

def graph_randomly_subject_sensor(metadata, data_lookup, zero_crossing_lookup):
  print("randomly selects a subject and a column to graph")

  inout ='outdoors'
  subjectID_list = metadata['subjectID'].unique()
  subjectID = subjectID_list[random.randint(0,len(subjectID_list)-1)]
  column_to_graph = COLUMNS_TO_GRAPH[random.randint(0,len(COLUMNS_TO_GRAPH)-1)]
  all_gates = aggregate_single_subject(data_lookup, metadata, zero_crossing_lookup, column_to_graph, inout , subjectID)
  avg = all_gates.mean(axis=0)
  std = all_gates.std(axis=0)
  fig, ax = plt.subplots()
  ax.plot(avg, color='black')
  label_axis(column_to_graph, ax)

  ax.fill_between(np.arange(0,100), avg-std, avg+std, alpha=0.4, color='gray')
  _= ax.set_title("subject " + str(subjectID) + " " +column_to_graph+ ' '+COLUMNS_TO_AREA[column_to_graph] + ' ' + inout) 

def each_subject_one_sensor(save_dir, data_lookup, metadata, zero_crossing_lookup, column_to_graph, inout, local_dir):
  assert len(metadata['pace'].unique())==1, "paces"+str(metadata['pace'].unique())
  for subjectID in  metadata['subjectID'].unique():
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5,5))
    filename = local_dir + "_"+str(subjectID)+".png"
    title = filename.replace(".png", "")
    all_gates = aggregate_single_subject(data_lookup, metadata, zero_crossing_lookup, column_to_graph, inout , subjectID)
    avg =all_gates.mean(axis=0)
    std = all_gates.std(axis=0)
    ax.plot(avg, color='black')
    ax.fill_between(np.arange(0,100), avg-std, avg+std, alpha=0.4, color='gray')
    _= ax.set_title(title )

    fig.savefig(os.path.join(save_dir,filename)  )
    plt.close()
def inout_each_subject(save_dir, data_lookup, metadata, zero_crossing_lookup, column_to_graph):
  for i, inout in enumerate(['indoors', 'outdoors']):
    local_dir = column_to_graph.replace("/", '-') + '_'+COLUMNS_TO_AREA[column_to_graph]+"_"+inout
    local_path = os.path.join(save_dir, local_dir)
    if not os.path.exists(local_path):
      os.mkdir(local_path)
    each_subject_one_sensor(local_path, data_lookup, metadata, zero_crossing_lookup, column_to_graph, inout, local_dir)

def each_sensor_each_subject(save_dir, data_lookup, metadata, zero_crossing_lookup):
  for column_to_graph in COLUMNS_TO_GRAPH:
    inout_each_subject(save_dir, data_lookup, metadata, zero_crossing_lookup, column_to_graph)


def save_sensor_data(save_dir, data_lookup, metadata, zero_crossing_lookup, column_to_graph, inout, df_columns):
  fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(7.5,5))#plt.subplots()
  df_sensor_subject = pd.DataFrame([], columns=df_columns) 
  global_avg = None
  for subjectID in  metadata['subjectID'].unique():

    all_gates = aggregate_single_subject(data_lookup, metadata, zero_crossing_lookup, column_to_graph, inout , subjectID)
    avg = all_gates.mean(axis=0)
    std = all_gates.std(axis=0)
    row = pd.DataFrame([[subjectID, 'avg', *avg ]], columns=df_columns)
    df_sensor_subject = pd.concat([df_sensor_subject,row ])
    row = pd.DataFrame([[subjectID,'std', *std ]], columns=df_columns)
    df_sensor_subject = pd.concat([df_sensor_subject,row ])
    ax.plot(avg, label=str(subjectID))
  _= ax.set_title(column_to_graph+ ' '+COLUMNS_TO_AREA[column_to_graph] + ' ' + inout) 
  filename = column_to_graph.replace("/", '-').replace(' ','_') + '_' + COLUMNS_TO_AREA[column_to_graph] + '_' + inout+'.csv'
  df_sensor_subject.to_csv(os.path.join(save_dir,filename), index=False)
  fig.savefig(os.path.join(save_dir,filename.replace('.csv', '.png')))
  plt.close()
  time.sleep(0.25)  

def save_each_subject_each_sensor(save_dir, data_lookup, metadata, zero_crossing_lookup):
  save_dir = os.path.join(save_dir, "data_each_subject_each_sensor")
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  df_columns = ['subjectID', 'avg-or-std', *[str(x) for x in range(100)]]
  for inout in ['outdoors', 'indoors']:
    for column_to_graph in COLUMNS_TO_GRAPH:
      save_sensor_data(save_dir, data_lookup, metadata, zero_crossing_lookup, column_to_graph, inout, df_columns)

def add_data_gate_peak_valley_swing(sensor, metadata, data_lookup, zero_crossing_lookup, data,subjectID,inout, area, leg):
  ##measure peak, valley, range, gate length all averages and std
  all_gates = aggregate_single_subject(data_lookup, metadata, zero_crossing_lookup, sensor, inout , subjectID)
  bounded_gates = (all_gates/abs(all_gates.min()))
  try:
    peaks, valleys, ranges = calculate_peaks_valley_range(bounded_gates.copy())
  except ValueError:
    print('sensor', sensor,"subject", subjectID, "inout", inout)
    raise ValueError
  #assign_peak_valley_each_file(df_peak_subject, data_lookup, sensor, zero_crossing_lookup)
  row = {'sensor':sensor, 'subjectID':subjectID, 'inout':inout, 'area':area,
              "avg_peak":peaks.mean(),"std_peak":peaks.std(),
                    "max_peak":peaks.max(), "min_peak":peaks.min(), 
              "avg_valley":valleys.mean(), "std_valley": valleys.std(), 
                    "max_valley":valleys.max(), "min_valley":valleys.min(), 
              "avg_range":ranges.mean(), "std_range":ranges.std(),                
                    'max_range':ranges.max(), 'min_range':ranges.min(),}
  if sensor in [RIGHT_AVY_HEADER, LEFT_AVY_HEADER]:
    
    swing_index = calculate_swing_index_gates(bounded_gates)
    if len(swing_index)!=len(bounded_gates): 
      print("error finding swing index")
      print("values used for swing index", len(swing_index), "values used for other calculations", len(bounded_gates))
      print("sensor {} subjectID {}".format(sensor,subjectID) )
      #swing_index = np.zeros(bounded_gates.shape[0])
    zero_crossings = get_zeros_crossings_single_subject(metadata, zero_crossing_lookup,subjectID, inout, leg)
    avg_gate_lengths,std_gate_lengths, max_gate_lengths, min_gate_lengths  = avg_std_gate_lengths(zero_crossings)
    row.update({'avg_gate_length':avg_gate_lengths,'std_gate_length':std_gate_lengths,
                  'max_gate_length':max_gate_lengths,'min_gate_length':min_gate_lengths,
                "avg_swing_index":swing_index.mean(), "std_swing_index":swing_index.std(), 
                  "max_swing_index":swing_index.max(), "min_swing_index":swing_index.min()
                  })
  data.append(row )

def gate_peak_valley_swing_save_data(columns_pkvgs, save_dir, sensor, metadata, data_lookup, zero_crossing_lookup ):
  assert len(metadata['pace'].unique())==1, "paces"+str(metadata['pace'].unique())
  pace=metadata['pace'].unique()[0]
  df_filename = sensor.replace("/", '-')+'_'+pace
  data = []
  leg = COLUMNS_TO_LEG[sensor]
  area = COLUMNS_TO_AREA[sensor]
  df_filename += "_"+area+'.csv'
  for i, inout in enumerate(['indoors', 'outdoors']):
    for subjectID in  metadata['subjectID'].unique():
      add_data_gate_peak_valley_swing(sensor, metadata, data_lookup, zero_crossing_lookup, data,subjectID,inout, area, leg)
  df_peak_subject = pd.DataFrame(data, columns=columns_pkvgs)
  df_peak_subject.to_csv(os.path.join(save_dir, df_filename), index=False)


def gate_peak_valley_swing(metadata, data_lookup, zero_crossing_lookup, SAVE_DIR):
  save_dir = os.path.join(SAVE_DIR, "peaks_per_subject")
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  all_mins = []
  columns_pkvgs = ['sensor', 'subjectID','inout', 'area',
            'avg_gate_length','std_gate_length', 'max_gate_length','min_gate_length',
            "avg_peak", "std_peak","max_peak", "min_peak", 
            "avg_valley", "std_valley","max_valley", "min_valley", 
            "avg_range", "std_range", 'max_range', 'min_range',
            "avg_swing_index", "std_swing_index", 
              "max_swing_index", "min_swing_index"]
  #avxyz_cols = []
  #for col in [RIGHT_AVY_HEADER, LEFT_AVY_HEADER]:
  #  avxyz_cols.extend([col, col.replace('Y', 'X'), col.replace('Y','Z')])
  print("grabbing data for data streams/columns")
  #print(avxyz_cols)
  for sensor in COLUMNS_TO_GRAPH:#[RIGHT_AVY_HEADER, LEFT_AVY_HEADER]:
    gate_peak_valley_swing_save_data(columns_pkvgs,save_dir, sensor, metadata, data_lookup, zero_crossing_lookup )


def get_filenames_single_subject(metadata, subjectID, inout):
  assert len(metadata['pace'].unique())==1, "paces"+str(metadata['pace'].unique())
  where_cond = ((metadata['subjectID']==subjectID)&(metadata['inout']==inout))
  records = metadata[where_cond]
  return records.filename.values
def get_zeros_crossings_single_subject(metadata,zero_crossing_lookup, subjectID, inout, leg):
    filenames_p_subject = get_filenames_single_subject(metadata, subjectID, inout)
    zero_crossings = []
    for filename in filenames_p_subject:
      if len(zero_crossing_lookup[filename][leg])==1:
        print(zero_crossing_lookup[filename][leg])
        raise ValueError('found one gate in entire file '+filename)
      zero_crossings.extend( zero_crossing_lookup[filename][leg])    
    if len(zero_crossings)==0:
      print("subject", subjectID, 'inout', inout)
      raise ValueError('found no gates for subject '+str(subjectID))
    return zero_crossings 

def calculate_swing_index_gates(all_gates):
  swing_index = []
  for i in range(all_gates.shape[0]):
    try:
      stance_change_index, stance_change_value = find_swing_stance_index(all_gates[i])  
    except ValueError:
      pass
    else:
      swing_index.append( stance_change_index)
  return np.array(swing_index)


def calculate_peaks_valley_range(all_gates):
  peaks = np.zeros(all_gates.shape[0])
  valleys = np.zeros(all_gates.shape[0])
  for i in range(all_gates.shape[0]):
    try:
      peaks[i] = max_peak(all_gates[i])[1]
    except ValueError:
       peaks[i] =all_gates[i].max()
    try:
      valleys[i] = find_lowest_valley(all_gates[i])[1]
    except ValueError:
      valleys[i] =all_gates[i].min()
  return peaks, valleys, peaks-valleys

def graph_avg_hist(save_dir, inout, fpath, avg_col, dstream):
  fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
  ax1.hist(dstream)
  fname = os.path.basename(fpath)
  ax1.set_title("histogram "+inout+" "+ fname)
  ax1.set_xlabel(avg_col + " one value for each subject")
  linetype  = 's'
  sm.qqplot(dstream, ax=ax2, line=linetype)
  ax2.set_title("quantile-quantile linetype:"+linetype)
  fig.savefig(os.path.join(save_dir,fname.replace(".csv","")+"_"+inout+".png" ))
  plt.close(fig)  
  return fname

def calc_wilcoxon(data_w, source, avg_col, df_p):
  alternative = 'two-sided'
  zero_method = 'wilcox'
  dstream_in = df_p[df_p.inout=='indoors'][avg_col].to_numpy()
  dstream_out = df_p[df_p.inout=='outdoors'][avg_col].to_numpy()
  w_statistic, p_value = wilcoxon(dstream_in, dstream_out,alternative=alternative,  zero_method= zero_method)
  data_w.append({'source':source, "data":avg_col, 
                  "w_statistic":w_statistic, "p_value":p_value, })
                  #"degrees_freedom":degrees_freedom, "confidence_interval "})
  return "alternative,{}\ntest,{}\na,indoors\nb,outdoors\nzero_method,{}\n".format(alternative, wilcoxon.__name__, zero_method)

def calc_t_test(data_t, source, avg_col, df_p):
  alternative = 'two-sided'
  ttest = ttest_rel
  dstream_in = df_p[df_p.inout=='indoors'][avg_col].to_numpy()
  dstream_out = df_p[df_p.inout=='outdoors'][avg_col].to_numpy()
  t_statistic, p_value = ttest(dstream_in, dstream_out,alternative=alternative )
  data_t.append({'source':source, "data":avg_col, 
                  "t_statistic":t_statistic, "p_value":p_value, })
                  #"degrees_freedom":degrees_freedom, "confidence_interval "})
  return "alternative,{}\ntest,{}\na,indoors\nb,outdoors\n".format(alternative, ttest.__name__)

def calc_shapiro(data_s,data_t, data_w, fpath, avg_cols, df_p, save_dir, alpha = 0.05):
  '''df_p is for a single sensor. each row is a different subject for indoors or outdoors
  alpha is the threshold for the p value to determine parametric vs non-parametric for t test or wilcoxon test
  '''
  for avg_col in avg_cols:
    for inout in ['indoors', 'outdoors']:
      pd_series = df_p[df_p.inout==inout][avg_col]
      if not pd_series.isnull().any():
        dstream = pd_series.to_numpy()
        fname = graph_avg_hist(save_dir, inout, fpath, avg_col, dstream)
        source = fname.replace(".csv","")
        time.sleep(0.1)
        w_statistic, p_value = shapiro(dstream)
        data_s.append({'source':source, "inout":inout, "data":avg_col, 
                       "w_statistic":w_statistic, "p_value":p_value,
                       "avg":dstream.mean(), 'std':dstream.std()})
        if inout=='indoors':
          p_indoors = p_value
        elif inout=='outdoors':
          p_outdoors = p_value
          if (p_indoors >alpha) and (p_outdoors> alpha):
            calc_t_test(data_t, source, avg_col, df_p)
          else:
            calc_wilcoxon(data_w, source, avg_col, df_p)
        
def calc_shapiro_t_test(SAVE_DIR):
  load_dir = os.path.join(SAVE_DIR, "peaks_per_subject")
  peak_files = glob.glob(os.path.join(load_dir,"*.csv"))
  save_dir = os.path.join(load_dir, "gaussian_analysis")
  data_s=[]
  data_t = []
  data_w = []
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  for fpath in peak_files:
    df_p = pd.read_csv(fpath)
    avg_cols = [x for x in df_p.columns if 'avg' in x]
    calc_shapiro(data_s, data_t, data_w, fpath, avg_cols, df_p, save_dir)
  
  df_s = pd.DataFrame(data_s, columns=['source', "inout", "data", "w_statistic", "p_value", 'avg', 'std'])
  df_s.to_csv(os.path.join(save_dir,"test_shapiro_wilk.csv" ))  

  df_t = pd.DataFrame(data_t, columns=['source', "data", "t_statistic", "p_value"])
  df_t.to_csv(os.path.join(save_dir,"test_t.csv" ))  
  df_w = pd.DataFrame(data_w, columns=['source', "data", "w_statistic", "p_value"])
  df_w.to_csv(os.path.join(save_dir,"test_wilcoxon.csv" ))  
  _= '''  with open(os.path.join(save_dir, "t_test_params.txt"), 'w') as fileobj:
        fileobj.write("parameters used in t test\n")
        fileobj.write(params)'''

def find_file_name_w_sensor(sensor:str, file_list: List[str]) -> str:
  for file in file_list:
    if sensor.replace("/", "-") in file:
      return file
  raise ValueError(f" couldn't find {sensor} in list of files")

def calc_shapiro_t_test_legs_combined(save_dir_nc):
  load_dir = os.path.join(save_dir_nc, "peaks_per_subject")
  peak_files = glob.glob(os.path.join(load_dir,"*.csv"))
  save_dir = os.path.join(load_dir, "gaussian_analysis")
  data_s=[]
  data_t = []
  data_w = []
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  for sensor_cols in COLUMNS_BY_SENSOR:
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(15,5))
    fig2, ax2 = plt.subplots(nrows=1,ncols=2, figsize=(15,5))
    sensor_name = sensor_cols['left'].replace(".1",' '+sensor_cols['sensor']).replace(".3",' '+sensor_cols['sensor'])
    left_file = find_file_name_w_sensor(sensor_cols['left'], peak_files)
    right_file = find_file_name_w_sensor(sensor_cols['right'], peak_files)
    comb_file_path = os.path.join(load_dir, sensor_name.replace("/","-"))
    ##comb_file_path = left_file.replace(sensor_cols['left'].replace("/","-"), sensor_name).replace("left", '').replace("right",)
    df_l = pd.read_csv(left_file)
    df_r = pd.read_csv(right_file)
    df_c = pd.concat([df_l, df_r])
    avg_cols = [x for x in df_c.columns if 'avg' in x]
    #print(comb_file_path)
    calc_shapiro(data_s, data_t, data_w, comb_file_path, avg_cols, df_c, save_dir)
  
  df_s = pd.DataFrame(data_s, columns=['source', "inout", "data", "w_statistic", "p_value", 'avg', 'std'])
  df_s['test_type'] = 'shapiro_wilk'
  df_s.to_csv(os.path.join(save_dir,"combined_legs_test_shapiro_wilk.csv" ))  

  df_t = pd.DataFrame(data_t, columns=['source', "data", "t_statistic", "p_value"])
  df_t['test_type'] = 't_test'
  #df_t.to_csv(os.path.join(save_dir,"test_t.csv" ))  
  df_w = pd.DataFrame(data_w, columns=['source', "data", "w_statistic", "p_value"])
  df_w['test_type']='wilcoxon'
  #df_w.to_csv(os.path.join(save_dir,"test_wilcoxon.csv" ))  
  df_stat = pd.concat([df_w, df_t])
  df_stat.to_csv(os.path.join(save_dir,"combined_legs_test_t_and_wilcoxon.csv"))
  _= '''  with open(os.path.join(save_dir, "t_test_params.txt"), 'w') as fileobj:
        fileobj.write("parameters used in t test\n")
        fileobj.write(params)'''


def aggregate_subjects_trials(data_lookup, metadata: pd.core.frame.DataFrame, zero_crossing_lookup: dict, column_to_graph:str, inout:str = 'indoors'):
  '''given one column to focus on, it combines all the data across all the files and subjects.
  inout could be either [indoors, outdoors]
  to combine different trials, group by 'subjectID', 'pace', 'inout', 
  then loop through all df's created in the group by iterable. in each df, 
  if more than one row exists, concatenate the data from the rows'''
  agg_gates  = None
  for subjectID in  metadata['subjectID'].unique():
    all_gates = aggregate_single_subject(data_lookup, metadata, zero_crossing_lookup, column_to_graph, inout , subjectID)
    if type(agg_gates)==np.ndarray:
     agg_gates = np.concatenate((agg_gates,all_gates),axis=0)
    else: 
     agg_gates=all_gates
  return agg_gates

def plot_columns_inout(data_lookup, metadata: pd.core.frame.DataFrame, zero_crossing_lookup: dict, column_to_graph:str, inout:str = 'indoors'):
  agg_gates=aggregate_subjects_trials(data_lookup, metadata, zero_crossing_lookup,column_to_graph, inout )
  avg = agg_gates.mean(axis=0)
  std = agg_gates.std(axis=0)
  fig, ax = plt.subplots()
  ax.plot(avg, color='black')
  ax.set_xlabel("normalized time")
  ax.set_ylabel("signal")
  label_axis(column_to_graph, ax)

  ax.fill_between(np.arange(0,100), avg-std, avg+std, alpha=0.4, color='gray')
  _= ax.set_title(column_to_graph+ ' '+COLUMNS_TO_AREA[column_to_graph] + ' ' + inout) 

def graph_sensors_combined_subjects_trials(save_dir, data_lookup, metadata, zero_crossing_lookup):
  '''graph all data streams indoors, then outdoors aggregated across all subjects 
     and trials. only aggregates across two trials. if more exist, they are ignored'''
  for column_to_graph in COLUMNS_TO_GRAPH:
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(15,5))
    for i, inout in enumerate(['indoors', 'outdoors']):
      agg_gates=aggregate_subjects_trials(data_lookup, metadata, zero_crossing_lookup,column_to_graph, inout )
      avg = agg_gates.mean(axis=0)
      std = agg_gates.std(axis=0)
      ax[i].plot(avg, color='black')
      label_axis(column_to_graph, ax[i])

      ax[i].fill_between(np.arange(0,100), avg-std, avg+std, alpha=0.4, color='gray')
      title = column_to_graph+ ' '+COLUMNS_TO_AREA[column_to_graph]
      _= ax[i].set_title(title + ' ' + inout)
    fig.savefig(os.path.join(save_dir,title.replace("/", '-')+'.png'))  

def combined_subjects_trials_signal_stats(data_lookup, metadata, zero_crossing_lookup, save_dir):
  print("all values presented are done by analyzing a list of size ~9,198. Where each element in the list is a gate profile. Meaning the average min is the average of ~9,198 minimum values")
  global_mins = []
  ranges = []
  range_std = []
  global_maxes = []
  storing_data = []
  for column_to_graph in COLUMNS_TO_GRAPH:
    for i, inout in enumerate(['indoors', 'outdoors']):
      agg_gates=aggregate_subjects_trials(data_lookup, metadata, zero_crossing_lookup,column_to_graph, inout )
      maxes_of_all_gates_trails = agg_gates.max(axis=1)
      mins_of_all_gates_trails = agg_gates.min(axis=1)
      ranges_of_all_gates_trails = np.subtract(maxes_of_all_gates_trails, mins_of_all_gates_trails)
      #print("")
      #print('\n',column_to_graph, COLUMNS_TO_AREA[column_to_graph], inout)
      global_min = mins_of_all_gates_trails.min()
      #print('\t', "global min" ,global_min )
      global_mins.append(global_min)
      avg_min = mins_of_all_gates_trails.mean()
      #print("\t","avg min",avg_min  )
      std_min = mins_of_all_gates_trails.std()
      #print("\t","std min", std_min )
      global_max = maxes_of_all_gates_trails.max()
      #print('\t', "global max" , global_max)
      global_maxes.append(global_max)
      avg_range = ranges_of_all_gates_trails.mean()
      #print('\t', "avg range" ,avg_range )
      ranges.append(avg_range)
      std_range = ranges_of_all_gates_trails.std()
      #print('\t', "std range" , std_range)
      range_std.append(std_range)
      number_of_gates = maxes_of_all_gates_trails.shape[0]
      #print("number of gates",number_of_gates)
      
      storing_data.append([column_to_graph, COLUMNS_TO_AREA[column_to_graph], inout,global_min, avg_min,std_min,global_max, avg_range, std_range,number_of_gates ])
  storing_data_df = pd.DataFrame(storing_data, columns=['column_to_graph', 'area', 'inout','global_min', 'avg_min','std_min','global_max', 'avg_range', 'std_range','number_of_gates'])
  storing_data_df.to_csv(os.path.join(save_dir, 'combined_subjects_trials_signal_stats.csv'))
  return global_mins, global_maxes, ranges

def combine_legs_flip(left_gates, right_gates, sensor_name):
  '''some signals are measured upside down to the other other side. negate them
  before combining'''
  TO_FLIP = ['Acceleration Y (m/s^2) thigh','Angular Velocity X (rad/s) shank', 
             'Angular Velocity X (rad/s) thigh','Angular Velocity Z (rad/s) thigh' ,
           'Acceleration Y (m/s^2) shank']
  if sensor_name in TO_FLIP:
    right_avg = (right_gates*-1).mean(axis=0)
    aggg_gates = np.concatenate([left_gates, (right_gates*-1)], axis=0)
  else:
    right_avg = (right_gates).mean(axis=0)
    aggg_gates = np.concatenate([left_gates, (right_gates)], axis=0)
  return right_avg, aggg_gates

def graph_each_leg_multiline(ax, inout, sensor_name, left_avg, right_avg):
  #right_avg, aggg_gates = combine_legs_flip(left_gates, right_gates, sensor_name)
  ax.plot(left_avg, label='left')
  ax.plot(right_avg, label='right')
  ax.set_xlabel("normalized time")
  ax.set_ylabel("signal")
  #ax.plot(aggg_gates.mean(axis=0), label='mean')
  title2 = sensor_name+ ' '+'each leg'
  _= ax.set_title(title2+ ' ' + inout) 
  ax.legend()

  return title2 #, aggg_gates.mean(axis=0), aggg_gates.std(axis=0)
def graph_comb_legs_avg(ax,inout,sensor_name, avg,std):
  ax.plot(avg, color='black')
  ax.set_xlabel("normalized time")
  ax.set_ylabel("signal")
  ax.fill_between(np.arange(0,100), avg-std, avg+std, alpha=0.4, color='gray')
  title = sensor_name+ ' '+'both legs'
  _= ax.set_title(title+ ' ' + inout) 
  return title

def aggregate_subjects_trials_legs(data_lookup, metadata: pd.core.frame.DataFrame, zero_crossing_lookup: dict):
  combined_legs = {}
  for sensor_cols in COLUMNS_BY_SENSOR:
    sensor_name = sensor_cols['left'].replace(".1",' '+sensor_cols['sensor']).replace(".3",' '+sensor_cols['sensor'])
    combined_legs[sensor_name] = {}
    for i, inout in enumerate(['indoors', 'outdoors']):
      left_gates, right_avg,avg, std = combine_left_right_legs(combined_legs, sensor_name, sensor_cols, i, inout, data_lookup, metadata, zero_crossing_lookup)
  return combined_legs

def combine_left_right_legs(combined_legs, sensor_name, sensor_cols, i, inout, data_lookup, metadata, zero_crossing_lookup):
  left_gates  =aggregate_subjects_trials(data_lookup, metadata, zero_crossing_lookup,sensor_cols['left'], inout )
  right_gates =aggregate_subjects_trials(data_lookup, metadata, zero_crossing_lookup,sensor_cols['right'], inout )
  
  right_avg, aggg_gates = combine_legs_flip(left_gates, right_gates, sensor_name)
  avg = aggg_gates.mean(axis=0)
  std = aggg_gates.std(axis=0)
  combined_legs[sensor_name][inout] = {'avg': avg, 'std':std}
  return left_gates, right_avg,avg, std

def graph_aggregate_subjects_trials_legs(save_dir, data_lookup, metadata: pd.core.frame.DataFrame, zero_crossing_lookup: dict):
  '''for each sensor, aggregate the signal from all subjects and trials and add 
  on the signals for each leg together. So that you get one signal for each 
  sensor for indoor/outdoor for both legs.
  also graph the mean of each leg seperately on the same plot so you can see 
  what they look like before the averaging.
  return the mean and std of the combined signal'''
  combined_legs = {}
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  for sensor_cols in COLUMNS_BY_SENSOR:
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(15,5))
    fig2, ax2 = plt.subplots(nrows=1,ncols=2, figsize=(15,5))
    sensor_name = sensor_cols['left'].replace(".1",' '+sensor_cols['sensor']).replace(".3",' '+sensor_cols['sensor'])
    combined_legs[sensor_name] = {}
    for i, inout in enumerate(['indoors', 'outdoors']):
      left_gates, right_avg,avg, std = combine_left_right_legs(combined_legs, sensor_name, sensor_cols, i, inout, data_lookup, metadata, zero_crossing_lookup)
      title = graph_comb_legs_avg(ax[i],inout,sensor_name,avg ,std)
      title2= graph_each_leg_multiline(ax2[i], inout, sensor_name, left_gates.mean(axis=0), right_avg)
      #title2, avg, std  = graph_each_leg_multiline(ax2[i], inout, sensor_name, left_gates, right_gates.copy())
      #ax[i].plot(avg)
      #ax[i].fill_between(np.arange(0,100), avg-std, avg+std, alpha=0.4, color='gray')

      #ax[i].set_title(inout+" "+"mean"+" " +sensor_name)
    fig.savefig(os.path.join(save_dir,title.replace("/", '-')+'.png'))
    fig2.savefig(os.path.join(save_dir,title2.replace("/", '-')+'.png'))
  return combined_legs


def cadence_per_subject(save_dir,metadata, zero_crossing_lookup, add_dir='stats_of_gate_lengths',leg ='left'):
  save_dir = os.path.join(save_dir, add_dir)
  ##data points per step to steps/minute
  if leg=='right':
    sensor = RIGHT_AVY_HEADER
  elif leg=='left':
    sensor=LEFT_AVY_HEADER
  df_cadence = pd.DataFrame()
  assert len(metadata['pace'].unique())==1, "paces"+str(metadata['pace'].unique())
  for subjectID in  metadata['subjectID'].unique():
    for inout in ['outdoors', 'indoors']:
      where_cond = ((metadata['subjectID']==subjectID)&(metadata['inout']==inout))
      records = metadata[where_cond]
      filename =records['filename'].iloc[0]
      zero_crossings = zero_crossing_lookup[filename][leg]
      if records.shape[0]==2:
        filename =records['filename'].iloc[1]
        zero_crossings.extend(zero_crossing_lookup[filename][leg])
      cadence = [60*FREQUENCY/(x[1]-x[0]) for x in zero_crossings]
      cadence = np.array(cadence)
      cmean, cstd, cmax, cmin =  cadence.mean() , cadence.std(), cadence.max(), cadence.min()    
      row = {"sensor":sensor, "leg":leg, "subjectID": subjectID, 'inout':inout,
             "cadence_avg_step_p_minute":cmean,"cadence_std":cstd,"cadence_max":cmax,"cadence_min":cmin  }
      df_cadence = pd.concat([df_cadence, pd.DataFrame([row])])
  df_cadence.to_csv(os.path.join(save_dir, "per_subject_"+sensor.replace("/", "-").replace(" ", '')+'_'+leg+".csv"))
  return df_cadence

def hist_cadence():
  save_dir = os.path.join("Analysis",'cadence')
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  
  for speed in ['slow', 'normal','fast']:
    lookup_dir = os.path.join("Analysis",speed, 'stats_of_gate_lengths')
    filename = glob.glob(os.path.join(lookup_dir,"per_subject_*.csv"))[0]
    print("looking at file", filename)
    df_cadence=pd.read_csv(filename)
    for inout in ['indoors', 'outdoors']:
      signal = df_cadence[df_cadence['inout']==inout]['cadence_avg_step_p_minute'].values
      title = "cadence_{}_{}".format(inout, speed)
      fig, ax1 = plt.subplots(1, 1, figsize=(8,6))
      ax1.hist(signal)
      ax1.set_title(title)
      ax1.set_xlabel("cadence")
      ax1.set_ylabel("ct subjects")
      ax1.grid(visible=True)
      fig.savefig(os.path.join(save_dir, title+'.png')) 

def line_plot_cadence():
  save_dir = os.path.join("Analysis",'cadence')
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  df_list = []
  for speed in ['slow', 'normal','fast']:
    lookup_dir = os.path.join("Analysis",speed, 'stats_of_gate_lengths')
    filename = glob.glob(os.path.join(lookup_dir,"per_subject_*.csv"))[0]
    print("looking at file", filename)
    df_cadence=pd.read_csv(filename)
    df_list.append(df_cadence)
  list_subjects = list(df_list[0]['subjectID'].unique())
  list_subjects.sort()
  print("list subjects slow:", list_subjects)
  print("list subjects medium:", sorted(list(df_list[1]['subjectID'].unique())))
  print("list subjects fast:", sorted(list(df_list[2]['subjectID'].unique())))
  x = np.array([0,1,2])
  lines_p_graph = 5
  ct_cols = len(list_subjects)//lines_p_graph
  fig, ax = plt.subplots(ct_cols, 2, figsize=(8,17))
  for i, inout in enumerate(['indoors', 'outdoors']):
    for pi in range(ct_cols):
      ax1 = ax[pi,i]
      title = "cadence_{}".format(inout)      
      ax1.set_title(title)
      ax1.set_ylabel("cadence")
      ax1.grid(visible=True)      
      for subjectID in list_subjects[pi*lines_p_graph:(pi+1)*lines_p_graph]:
        y = []
        for df in df_list:
          y.append(df[(df['subjectID']==subjectID)&(df['inout']==inout)]['cadence_avg_step_p_minute'].iloc[0])
        ax1.plot(x,y, label=str(subjectID))
      ax1.set_xticks(x, ['slow', 'normal','fast'])
      ax1.legend()
    #fig.savefig(os.path.join(save_dir, title+'.png')) 


def cadence_remove_outlier(base_dir):
  outliers = set()
  for speed in ['slow', 'normal','fast']:
    lookup_dir = os.path.join(base_dir,speed, 'stats_of_gate_lengths')
    filename = glob.glob(os.path.join(lookup_dir,"per_subject_*.csv"))[0]
    print("looking at file", filename)
    df_cadence=pd.read_csv(filename)
    for inout in ['indoors', 'outdoors']:
      list_subjects = list(df_cadence['subjectID'].unique())
      signal = df_cadence[df_cadence['inout']==inout]['cadence_avg_step_p_minute'].values
      std = signal.std()
      avg = signal.mean()
      new_signal = []
      for subj in list_subjects:
        val = df_cadence[(df_cadence['inout']==inout) & (df_cadence['subjectID']==subj)]['cadence_avg_step_p_minute'].iloc[0]
        if abs(avg-val)<std:
          new_signal.append(val)
        else:
          print(inout,speed, subj, '\t', val, avg, std)
          outliers.add(subj) 

      print(inout, speed, round(np.array(new_signal).mean(),2))
  with open(os.path.join(base_dir, 'cadence_outliers.pickle'), 'wb')as fileobj:
    pickle.dump(outliers,fileobj)
  return outliers
########################################################################################
########################################################################################

def one_subject_summary(subjectID):
  rows = []
  for speed in ['slow', 'normal','fast']:
    lookup_dir = os.path.join("Analysis",speed, 'stats_of_gate_lengths')
    filename = glob.glob(os.path.join(lookup_dir,"per_subject_*.csv"))[0]
    print("looking at file", filename)
    df_cadence=pd.read_csv(filename)
    indoors = df_cadence[(df_cadence['inout']=='indoors')&(df_cadence['subjectID']==subjectID)]['cadence_avg_step_p_minute'].iloc[0]
    outdoors = df_cadence[(df_cadence['inout']=='outdoors')&(df_cadence['subjectID']==subjectID)]['cadence_avg_step_p_minute'].iloc[0]
    rows.append([speed, indoors, outdoors, subjectID])
  df = pd.DataFrame(rows, columns=['pace', 'indoors', 'outdoors', 'subjectID'])
  df.to_csv(os.path.join("Analysis", 'cadence', 'subjectID_'+str(subjectID)+'.csv'))
  return df


def pull_together_across_speeds():
  '''pulls the cadence values together across speeds'''
  cadence_dir =os.path.join("Analysis", 'cadence')
  if not os.path.exists(cadence_dir):
    os.mkdir(cadence_dir)
  compare_cadence = []
  for speed in ['slow', 'normal','fast']:
   lookup_dir = os.path.join("Analysis",speed, 'stats_of_gate_lengths')
   filename = glob.glob(os.path.join(lookup_dir,"per_subject_*.csv"))[0]
   print("looking at file", filename)
   df_cadence=pd.read_csv(filename)
   indoors = df_cadence[df_cadence['inout']=='indoors']['cadence_avg_step_p_minute'].values
   outdoors = df_cadence[df_cadence['inout']=='outdoors']['cadence_avg_step_p_minute'].values
   compare_cadence.append([speed, indoors.mean(), indoors.std(), outdoors.mean(), outdoors.std()])
  df_compare = pd.DataFrame(compare_cadence, columns=['pace', 'indoor mean step/minute', 'indoor std', 'outdoor mean', 'outdoor std']) 
  islow, inormal, ifast = df_compare['indoor mean step/minute'].values
  df_compare['indoors:abs(normal-x)/normal'] = [abs(inormal - islow)/inormal, 0, abs(inormal-ifast)/inormal]
  oslow, onormal, ofast = df_compare['outdoor mean'].values
  df_compare['outdoors:abs(normal-x)/normal'] = [abs(onormal - oslow)/onormal, 0, abs(onormal-ofast)/onormal]
  df_compare.to_csv(os.path.join("Analysis", 'cadence', "cadence_aggregate.csv"))
  return df_compare
########################################################################################
########################################################################################

def graph_poincare_per_leg(data_lookup, metadata, zero_crossing_lookup, sensor, save_dir):
  ''' for single sensor, for indoors and out, for each subject do poincare on the
  avg signal '''
  area=COLUMNS_TO_AREA[sensor]
  save_dir = os.path.join(save_dir,sensor.replace("/", "-").replace(" ", '')+'_'+area.replace(' ',"_"))
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  data_poincare=[]
  print("pace",metadata['pace'].unique())
  for inout in ['indoors', 'outdoors']:
    for subjectID in  metadata['subjectID'].unique():
      all_gates = aggregate_single_subject(data_lookup, metadata, zero_crossing_lookup, sensor, inout , subjectID)
      avg =all_gates.mean(axis=0)
      title = "{} {} {} subjectID: {}".format(sensor,area, inout,subjectID)
      figname = "{}_{}_{}_{}.png".format(sensor.replace("/", "-"),area, inout,subjectID)
      results = poincare(avg, title=title, show=False)
      fig = results['poincare_plot']
      columns = ['sensor','area','inout', 'subjectID', 'sd1', 'sd2', 'sd_ratio', 'ellipse_area']
      data_poincare.append([sensor,area, inout,subjectID, 
                            results['sd1'], results['sd2'], results['sd_ratio'], 
                            results['ellipse_area']])
      fig.savefig(os.path.join(save_dir, figname))
      plt.close()
      time.sleep(0.1)
  df_poincare = pd.DataFrame(data_poincare, columns=columns)
  df_poincare.to_csv(os.path.join(save_dir,'poincare_sd.csv'), index=False)  


def graph_poincare_comb_leg_per_sensor(combined_legs, sensor, save_dir_m, ylim=None, xlim=None):
  ''' for single sensor - combined across legs, for indoors and out, aggregate 
  across all subjects do poincare on the avg signal '''
  save_dir = os.path.join(save_dir_m,sensor.replace("/", "-").replace(" ", '').replace("^",'-'))
  #print("saving at",save_dir)
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  data_poincare=[]
  for inout in ['indoors', 'outdoors']:
    avg_signal = combined_legs[sensor][inout]['avg']
    title = "{} {}".format(sensor, inout)
    figname = "{}_{}.png".format(sensor.replace("/", "-"), inout)
    results = poincare(avg_signal, title=title, show=False, ylim=ylim, xlim=xlim)
    fig = results['poincare_plot']
    columns = ['sensor','inout', 'sd1', 'sd2', 'sd_ratio', 'ellipse_area']
    data_poincare.append([sensor, inout, 
                          results['sd1'], results['sd2'], results['sd_ratio'], 
                          results['ellipse_area']])
    fig.savefig(os.path.join(save_dir, figname))
    plt.close()
    time.sleep(0.1)
  df_poincare = pd.DataFrame(data_poincare, columns=columns)
  df_poincare.to_csv(os.path.join(save_dir,'poincare_sd.csv'), index=False)  
def graph_poincare_comb_leg(combined_legs, save_dir_m):
  print("pace",metadata['pace'].unique()[0])
  for sensor in list(combined_legs.keys()):
    graph_poincare_comb_leg_per_sensor(combined_legs, sensor, save_dir_m)

def graph_poincare_comb_leg_per_sensor_per_subject(metadata,data_lookup,zero_crossing_lookup, sensor_cols, save_dir_m ,ylim=None, xlim=None):
  ''' for single sensor - combined legs, for indoors and out, for each subject do poincare on the
  avg signal '''  
  sensor_name = sensor_cols['left'].replace(".1",' '+sensor_cols['sensor']).replace(".3",' '+sensor_cols['sensor'])
  save_dir = os.path.join(save_dir_m,sensor_name.replace("/",'-'))
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)  
  rows = []
  for subjectID in  metadata['subjectID'].unique():
      indoors, outdoors = combine_legs_single_subject(sensor_name,data_lookup, metadata, zero_crossing_lookup, sensor_cols, subjectID )
      for signal, inout in [(indoors,'indoors'), (outdoors, 'outdoors')]:
        title = "{} {} subject:{}".format(sensor_name, inout, subjectID)
        figname = "{}_{}_subject{}.png".format(sensor_name.replace("/",'-'), inout, subjectID)
        
        results = poincare(signal, title=title, show=False, ylim=ylim, xlim=xlim)
        fig = results['poincare_plot']
        rows.append({"sensor":sensor_name, "subjectID":subjectID, "inout":inout, 
                     'sd1':results['sd1'] ,"sd2":results['sd2'], 
                     "sd_ration":results['sd_ratio'], 
                     "ellipse_area":results['ellipse_area'],
                          })
        fig.savefig(os.path.join(save_dir, figname))
        plt.close()
        time.sleep(0.1)
  return rows
def graph_poincare_comb_leg_per_subject(metadata,data_lookup,zero_crossing_lookup, save_dir):
  '''loops through all the sensors to call graph_poincare_comb_leg_per_sensor_per_subject'''
  print('pace',metadata['pace'].unique())
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  df_poincare_p_subject= pd.DataFrame()
  for sensor_cols in COLUMNS_BY_SENSOR:
    #print(sensor_name)
    rows =graph_poincare_comb_leg_per_sensor_per_subject(metadata,data_lookup,zero_crossing_lookup, sensor_cols, save_dir)
    df_poincare_p_subject = pd.concat([df_poincare_p_subject, pd.DataFrame(rows)])
  df_save_path = os.path.join(save_dir, "poincare_p_subject.csv")
  df_poincare_p_subject.to_csv(df_save_path, index=False)
  print(df_save_path)
  return df_poincare_p_subject

def get_sub_poincare(df_poincare_p_subject, sensor, inout, col):
  sub = df_poincare_p_subject[(df_poincare_p_subject['sensor']==sensor )& (df_poincare_p_subject['inout']==inout) ]
  return sub[col].values
def poincare_sim_stats_per_sensor(SAVE_DIR, alpha = 0.05):
  '''using sd1 and sd2 to calculate sim stats for each sensor
  must be run after graph_poincare_comb_leg_per_subject to load the data for each 
  subject indoor and outdoor so that can be fed into the similarity calc'''
  lookup_dir = os.path.join(SAVE_DIR, 'poincare',"per_subject")
  lookup_df_path = os.path.join(lookup_dir,"poincare_p_subject.csv")
  print(lookup_df_path)
  df_poincare_p_subject = pd.read_csv(lookup_df_path)
  poin_stats_data = []
  for sensor in list(df_poincare_p_subject['sensor'].unique()):
    for col in ['sd1', 'sd2']:
      indoors = get_sub_poincare(df_poincare_p_subject, sensor, 'indoors', col)
      outdoors = get_sub_poincare(df_poincare_p_subject, sensor, 'outdoors', col)
      in_row, out_row = indoor_outdoor_similarity(indoors, outdoors, col, alpha)
      in_row['sensor'] = sensor
      out_row['sensor'] = sensor
      poin_stats_data.extend([in_row, out_row])  
  poin_stats = pd.DataFrame(poin_stats_data)
  poin_stats.to_csv(os.path.join(lookup_dir, 'poincare_sim_stats_per_sensor.csv'), index=False)

def poincare_z_score_cohens_d(indoors, outdoors):
  indoors_std = indoors.std()
  outdoors_mean=outdoors.mean()
  indoors_mean= indoors.mean()
  #print("indoor mean", indoors_mean, "std", indoors_std)
  z_score =  (outdoors_mean-indoors_mean)/indoors_std
  cohens_d = (outdoors_mean-indoors_mean)/np.sqrt((indoors_std**2+outdoors.std()**2)/2)
  return z_score, cohens_d
def load_poincare_data(save_dir):
  list_dirs = [x for x in os.listdir(save_dir) if 'left' not in x]
  list_dirs = [x for x in list_dirs if 'right' not in x]
  list_dirs = [x for x in list_dirs if os.path.isdir(os.path.join(save_dir,x))]
  print("dirs looking in", save_dir)
  print(list_dirs)
  all_poincare_stats = pd.DataFrame()
  for dr in list_dirs:
    csv = glob.glob(os.path.join(save_dir, dr, "*.csv"))
    all_poincare_stats = pd.concat([all_poincare_stats, pd.read_csv(csv[0])])  
  return all_poincare_stats


def indoor_outdoor_similarity(indoors, outdoors, col, alpha):
  z_score, cohens_d=poincare_z_score_cohens_d(indoors, outdoors)
  shapiro_statistic_indoors, p_indoors = shapiro(indoors)
  shapiro_statistic_outdoors, p_outdoors = shapiro(outdoors)
  if (p_indoors >alpha) and (p_outdoors> alpha):
    test_stat, p_value=calc_t_test_poincare(indoors, outdoors)
    test_type='t_test'
  else:
    test_stat, p_value=calc_wilcoxon_poincare(indoors, outdoors)
    test_type='wilcoxon'
  indoor_row = {'source':col,'inout':'indoors', 'avg':indoors.mean(), 'std':indoors.std(), 'shapiro_stat':shapiro_statistic_indoors,
                'shapiro_p_val':p_indoors}
  outdoor_row = {'source':col,'inout':'outdoors', 'avg':outdoors.mean(), 'std':outdoors.std(), 'shapiro_stat':shapiro_statistic_outdoors,
                'shapiro_p_val':p_outdoors, 'test_type':test_type, 'stat':test_stat, 'p_value':p_value, 
                'z_score':z_score, "cohens_d":cohens_d}
  return [indoor_row, outdoor_row]

def calc_wilcoxon_poincare(indoors, outdoors):
  alternative = 'two-sided'
  zero_method = 'wilcox'
  w_statistic, p_value = wilcoxon(indoors, outdoors,alternative=alternative,  zero_method= zero_method)
  return w_statistic, p_value 
def calc_t_test_poincare(indoors, outdoors):
  alternative = 'two-sided'
  ttest = ttest_rel
  t_statistic, p_value = ttest(indoors, outdoors,alternative=alternative )
  return t_statistic, p_value
def calculate_poincare_stats(alpha = 0.05):
  '''calc poincare using the values from each signal as the data for the lists
  to compare indoor and outdoor'''
  save_dir = os.path.join(SAVE_DIR, 'poincare')
  all_poincare_data= load_poincare_data(save_dir)
  poin_stats_data = []
  for col in ['sd1', 'sd2']:
    indoors = all_poincare_data[all_poincare_data['inout']=='indoors'][col].values
    outdoors = all_poincare_data[all_poincare_data['inout']=='outdoors'][col].values
    poin_stats_data.extend(indoor_outdoor_similarity(indoors, outdoors, col, alpha))
  poin_stats = pd.DataFrame(poin_stats_data)
  poin_stats.to_csv(os.path.join(save_dir, 'poincare_sim_stats.csv'), index=False)
  #print("indoor mean", indoors.mean(), "std", indoors.std())
  return poin_stats
########################################################################################
########################################################################################

def re_format_poincare_sim_stats(base_dir = os.path.join('Analysis')):
  df=pd.DataFrame()
  for speed in ['normal', 'slow', 'fast']:
    load_dir = os.path.join(base_dir, speed,'poincare', 'per_subject')
    df_p = pd.read_csv(os.path.join(load_dir, 'poincare_sim_stats_per_sensor.csv'))
    df_p['pace']=speed
    df = pd.concat([df, df_p])
  sensors = list(df['sensor'].unique())
  new_table = []
  for sensor in sensors:
    subdf = df[(df['sensor']==sensor)&(df['inout']=='outdoors')]
    row = [sensor]
    column_names = ['sensor']
    for speed in ['normal', 'slow', 'fast']:
      for sd in ['sd1', 'sd2']:
        for metric in ['p_value', 'z_score']:
          column_names.append(speed +' '+sd+' '+metric)
          cond = (subdf['source']==sd)&(subdf['pace']==speed)
          val=subdf[cond][metric].iloc[0]
          row.append(val)
    new_table.append(row)
  new_table= pd.DataFrame(new_table, columns=column_names)
  new_table.to_csv(os.path.join(base_dir, 'trends_across_pace', 'poincare_sim_stats.csv'), index=False)

def re_formatting_peak(base_dir = os.path.join('Analysis')):
  df=pd.DataFrame()
  for speed in ['normal', 'slow', 'fast']:
    load_dir = os.path.join(base_dir, speed,'peaks_per_subject', 'gaussian_analysis')
    df_p = pd.read_csv(os.path.join(load_dir, 'combined_legs_test_t_and_wilcoxon.csv'))
    df_p['pace']=speed
    df = pd.concat([df, df_p])
  sensors = list(df['source'].unique())
  new_table = []
  for sensor in sensors:
    row = [sensor]
    column_names = ['sensor']
    subdf = df[(df['source']==sensor)]
    for speed in ['normal', 'slow', 'fast']:
      for metric in ['avg_peak',  'avg_range','avg_valley']:
        cond = (subdf['data']==metric)&(subdf['pace']==speed)
        column_names.append(speed+'_'+metric)
        assert len(subdf[cond]['p_value'].to_numpy())==1
        row.append(subdf[cond]['p_value'].iloc[0])
    new_table.append(row)
  new_table= pd.DataFrame(new_table, columns=column_names)
  new_table.to_csv(os.path.join(base_dir,'trends_across_pace', 'peak_valley_sim_stats.csv'), index=False)

  new_table = []
  sensors = list(df[df['data']=='avg_gate_length']['source'].unique())
  for sensor in sensors:
    row = [sensor]
    column_names = ['sensor']
    subdf = df[(df['source']==sensor)]
    for speed in ['normal', 'slow', 'fast']:
      for metric in ['avg_swing_index','avg_gate_length']:
        cond = (subdf['data']==metric)&(subdf['pace']==speed)
        column_names.append(speed+'_'+metric)
        assert len(subdf[cond]['p_value'].to_numpy())==1
        row.append(subdf[cond]['p_value'].iloc[0])
    new_table.append(row)
  new_table= pd.DataFrame(new_table, columns=column_names)
  new_table.to_csv(os.path.join(base_dir,'trends_across_pace', 'gate_swing_sim_stats.csv'), index=False)

def re_format_distance_sim(base_dir = os.path.join('Analysis')):
  df=pd.DataFrame()
  for speed in ['normal', 'slow', 'fast']:
    load_dir = os.path.join(base_dir, speed,'combined_subjects_and_trials_and_legs')
    df_p = pd.read_csv(os.path.join(load_dir, 'signal_similarity_combined_legs.csv'))
    df_p['pace']=speed
    df = pd.concat([df, df_p])
  sensors = list(df['sensor'].unique())
  new_table = []
  for sensor in sensors:
    subdf = df[(df['sensor']==sensor)]
    row = [sensor]
    column_names = ['sensor' ]
    for speed in ['normal', 'slow', 'fast']:
      assert len(subdf[subdf['pace']==speed]['cosine_similarity'].to_numpy())==1
      row.extend([subdf[subdf['pace']==speed]['cosine_similarity'].iloc[0], subdf[subdf['pace']==speed]['euclidean_distance'].iloc[0]])
      column_names.extend([speed+'_cosine', speed+'_euclidean'])

    new_table.append(row)
  new_table= pd.DataFrame(new_table, columns=column_names)
  new_table.to_csv(os.path.join(base_dir,'trends_across_pace', 'cos_euclidean_sim_stats.csv'), index=False)

def re_format_paired_comparison(base_dir = os.path.join('Analysis')):
  df=pd.DataFrame()
  for speed in ['normal', 'slow', 'fast']:
    load_dir = os.path.join(base_dir, speed,'similarity_per_subject')
    df_p = pd.read_csv(os.path.join(load_dir, 'lr_control_compare_lio_rio.csv'))
    df_p['pace']=speed
    df = pd.concat([df, df_p])
  sensors = list(df['sensor'].unique())
  new_table = []
  data_columns = [ 'lrc_i.vs.o_cosine_similarity_test_p_value',
                  'lrc_i.vs.o_cosine_similarity_z_score', 
                  'lrc_i.vs.o_euclidean_distance_test_p_value',  
                  'lrc_i.vs.o_euclidean_distance_z_score']
  for sensor in sensors:
    subdf = df[(df['sensor']==sensor)]
    row = [sensor]
    column_names = ['sensor' ]
    for speed in ['normal', 'slow', 'fast']:
      for head in data_columns:
        assert len(subdf[subdf['pace']==speed][head].to_numpy())==1,subdf[head].to_numpy() 
        row.append(subdf[subdf['pace']==speed][head].iloc[0])
        column_names.append(speed+'_'+head)
    new_table.append(row)
  new_table= pd.DataFrame(new_table, columns=column_names)
  new_table.to_csv(os.path.join(base_dir,'trends_across_pace', 'paired_comparison_sim_stats.csv'), index=False)
########################################################################################
########################################################################################

def continuous_gate_crossings(file):
  ##raw data
  ##finding gate crossings without butterworth
  df_raw = load_data(file, low_pass = False)
  zero_crossings_right = extract_gate_crossings(df_raw,RIGHT_AVY_HEADER, gate_crossing= GATE_CROSSING)
  zero_crossings_left = extract_gate_crossings(df_raw,LEFT_AVY_HEADER, gate_crossing= GATE_CROSSING)
  ##conver zero crossings to tuples
  zero_crossings_right = pair_gate_ends(zero_crossings_right, sensor = RIGHT_AVY_HEADER,filename = file, gate_length_bounds=None)
  zero_crossings_left = pair_gate_ends(zero_crossings_left, sensor =LEFT_AVY_HEADER, filename = file, gate_length_bounds=None)
  ##check that the gates have the right shape
  zero_crossings_right = check_shape_zero_crossings(zero_crossings_right, df_raw[RIGHT_AVY_HEADER].values)
  zero_crossings_left = check_shape_zero_crossings(zero_crossings_left, df_raw[LEFT_AVY_HEADER].values)
  return zero_crossings_left,zero_crossings_right , df_raw

def raw_summary_graph_multi_gate(subjectID, sensor, gate_ind, zero_crossings, df_raw):
  ##graphing first gate crossing
  start_gate, end_gate = zero_crossings[gate_ind][0], zero_crossings[gate_ind+5][1]
  print('indices', start_gate, end_gate)
  points_raw = df_raw[sensor].values[start_gate:end_gate]
  t = np.array([x for x in range(len(points_raw))])/FREQUENCY
  fig, ax1 = plt.subplots(1, 1, figsize=(12,8))
  ax1.plot(t, points_raw, color='black')
  ax1.set_title('Raw signal')
  ax1.set_xlabel("time (s)")
  ax1.set_ylabel("rad/s")
  _=fig.suptitle("Subject " +str(subjectID)+" " + sensor+ " " + COLUMNS_TO_AREA[sensor])
  ax1.grid(visible=True)
  sensor = sensor.replace("/", "-")
  fig.savefig(os.path.join(SUMMARY_DIR, f'subject_{subjectID}_0rawmultigate_{sensor}.png')) 

def raw_summary_graph(subjectID, sensor, zero_crossings, df_raw):
  ##graphing first gate crossing
  start_gate, end_gate = zero_crossings[0]
  print('indices', start_gate, end_gate)
  points_raw = df_raw[sensor].values[start_gate:end_gate]
  t = np.array([x for x in range(len(points_raw))])/FREQUENCY
  fig, ax1 = plt.subplots(1, 1, figsize=(12,8))
  ax1.plot(t, points_raw, color='black')
  ax1.set_title('Raw signal')
  ax1.set_xlabel("time (s)")
  ax1.set_ylabel("rad/s")
  _=fig.suptitle("Subject " +str(subjectID)+" " + sensor+ " " + COLUMNS_TO_AREA[sensor])
  ax1.grid(visible=True)
  sensor = sensor.replace("/", "-")
  fig.savefig(os.path.join(SUMMARY_DIR, f'subject_{subjectID}_0raw_{sensor}.png'))  

def graph_summary_butterworth(subjectID,file, sensor, gate_ind, df_filtered, side, N=4, Wn=20):
  zero_crossings = zero_crossing_lookup[file][side]
  start_gate, end_gate = zero_crossings[gate_ind]
  print('indices', start_gate, end_gate)
  points_filtered = df_filtered[sensor].values[start_gate:end_gate]
  t = np.array([x for x in range(len(points_filtered))])/FREQUENCY

  fig, ax2 = plt.subplots(1, 1, figsize=(12,8))
  ax2.plot(t, points_filtered, color='black')
  ax2.set_xlabel("time (s)")
  ax2.set_ylabel("rad/s")
  ax2.set_title(" Butterworth Filter " +" N = "+str(N)+", Wn = "+str(Wn)+"Hz")
  _=fig.suptitle("Subject " +str(subjectID)+" " + sensor+ " " + COLUMNS_TO_AREA[sensor])
  ax2.grid(visible=True)
  sensor = sensor.replace("/", "-")
  fig.savefig(os.path.join(SUMMARY_DIR, f'subject_{subjectID}_1butterworth_{sensor}.png'))

def graph_gate_crossing_summary(subjectID, sensor, df_filtered, gate_ind, zero_crossings):
  start_gate, end_gate = zero_crossings[gate_ind][0], zero_crossings[gate_ind+5][1]
  filtered_values = df_filtered[sensor].values
  print('indices', start_gate, end_gate)
  points_gates=filtered_values[start_gate:end_gate]
  t = np.array([x for x in range(len(points_gates))])/FREQUENCY

  ct_gate_crossing_marks = 5
  y_index_points = [zero_crossings[gate_ind+i][1] for i in range(ct_gate_crossing_marks)]
  x_index_points = [zero_crossings[gate_ind+i][1]-zero_crossings[gate_ind][0] for i in range(ct_gate_crossing_marks)]
  x_index_points[-1] = x_index_points[-1] -1
  x_points = [t[i] for i in x_index_points]
  y_points = [filtered_values[i] for i in y_index_points]

  fig, ax3 = plt.subplots(1, 1, figsize=(12,8))
  ax3.plot(t, points_gates, color='black')
  ax3.scatter(x_points, y_points, color='r', s=200)
  ax3.set_xlabel("time (s)")
  ax3.set_ylabel("rad/s")
  ax3.set_title(" Gate Detection ")
  _=fig.suptitle("Subject " +str(subjectID)+" " + sensor+ " " + COLUMNS_TO_AREA[sensor])
  ax3.grid(visible=True)
  sensor = sensor.replace("/", "-")
  fig.savefig(os.path.join(SUMMARY_DIR, f'subject_{subjectID}_2gate_detection_{sensor}.png'))  

def graph_summary_avg(subjectID, sensor, indoors):
  all_gates = aggregate_single_subject(data_lookup, metadata, zero_crossing_lookup, sensor, indoors , subjectID)
  avg = all_gates.mean(axis=0)
  std = all_gates.std(axis=0)
  fig, ax = plt.subplots(figsize=(12,8))
  ax.plot(avg, color='black')
  label_axis(sensor, ax)
  ax.grid(visible=True)
  ax.fill_between(np.arange(0,100), avg-std, avg+std, alpha=0.4, color='gray')
  ax.set_title(" Averaged Signal ")
  _=fig.suptitle("Subject " +str(subjectID)+" " + sensor+ " " + COLUMNS_TO_AREA[sensor])
  sensor = sensor.replace("/", "-")
  fig.savefig(os.path.join(SUMMARY_DIR, f'subject_{subjectID}_3averaged_signal_{sensor}.png'))   
  return avg, std

def graph_summary_avg_marked(subjectID, sensor, avg, std):
  fig, ax = plt.subplots(figsize=(12,8))
  ax.plot(avg, color='black')
  tup1, tup2 = max_peak(avg, edge_start=3), find_lowest_valley(avg)
  ax.scatter([tup1[0], tup2[0]],[tup1[1], tup2[1]], color='r', s=200)
  label_axis(sensor, ax)
  ax.grid(visible=True)
  ax.fill_between(np.arange(0,100), avg-std, avg+std, alpha=0.4, color='gray')
  ax.set_title(" Averaged Signal ")
  _=fig.suptitle("Subject " +str(subjectID)+" " + sensor+ " " + COLUMNS_TO_AREA[sensor])
  sensor = sensor.replace("/", "-")
  fig.savefig(os.path.join(SUMMARY_DIR, f'subject_{subjectID}_4labeled_peak_valley_{sensor}.png'))   

def make_summary_plots(metadata, subjectID =None, sensor=None):
  if not subjectID:
    subjectID = metadata['subjectID'].values[random.randint(0,metadata.shape[0]-1)]
  #'20221030-100150-subject_21inw1.csv'
  file =metadata['filename'][(metadata['subjectID']==subjectID) & (metadata['inout']=='indoors') & (metadata['trial']==1)].iloc[0]
  subjectID, indoors, speed , _, _  = extract_trial_data(file)#
  print("file", file)
  print("subject", subjectID)
  print("inout", indoors, "speed", speed) 
  if not sensor:
    sensor = LEFT_AVY_HEADER

  print("sensor", sensor)
  zero_crossings_left,zero_crossings_right, df_raw = continuous_gate_crossings(file)
  side = COLUMNS_TO_LEG[sensor]
  if side=='right':
    zero_crossings=zero_crossings_right
  elif side=='left':
    zero_crossings=zero_crossings_left
  gate_ind = np.random.randint(1,(len(zero_crossing_lookup[file][side])-1)) # random.randint(0,len(zero_crossings)-2)  

  ##filtered data
  ##zero crossings loaded from pickle
  ##graphing first gate crossing
  raw_summary_graph_multi_gate(subjectID, sensor,gate_ind, zero_crossings, df_raw)
  #raw_summary_graph(subjectID, sensor, zero_crossings, df_raw)
  #graph_summary_butterworth(subjectID,file, sensor, gate_ind, data_lookup[file], side)
  graph_gate_crossing_summary(subjectID, sensor, data_lookup[file], gate_ind,zero_crossing_lookup[file][side] )
  avg, std = graph_summary_avg(subjectID, sensor, indoors)
  graph_summary_avg_marked(subjectID, sensor, avg, std)

########################################################################################
########################################################################################

def get_signal_indoors_outdoors(data_lookup, metadata, zero_crossing_lookup, column_to_graph):
  '''returns: np array of avg signal'''
  agg_gates=aggregate_subjects_trials(data_lookup, metadata, zero_crossing_lookup,column_to_graph, 'indoors' )
  signal_indoors = agg_gates.mean(axis=0)
  agg_gates=aggregate_subjects_trials(data_lookup, metadata, zero_crossing_lookup,column_to_graph, 'outdoors' )
  signal_outdoors = agg_gates.mean(axis=0)  
  return signal_indoors, signal_outdoors

def normalize_signal_by_range(indoor, outdoor):
  '''takes in indoor and outdoor signals and divides all numbers by the range
  the purpose is s.t. the signal strength won't impact the result of the 
  cross correlation since the goal of the cross correlation is s.t.
  it compares signal shapes, not amplitudes'''
  inrange = indoor.max()-indoor.min()
  outrange = outdoor.max()-outdoor.min()
  return indoor/inrange, outdoor/outrange

def measure_correlation(signal_indoors, signal_outdoors, verbose=False):
  corr = correlate(signal_indoors, signal_outdoors)
  peak_corr = max_peak(corr)
  auto_corr = correlate(signal_indoors, signal_indoors)
  peak_auto_corr = max_peak(auto_corr)
  corr_delay = peak_corr[0]/2 - peak_auto_corr[0]/2
  normalized_correlation = peak_corr[1]/ peak_auto_corr[1]
  if verbose:
    print("\nsensor:", column_to_graph, COLUMNS_TO_AREA[column_to_graph])
    print(" correlation peak divided by auto-correlation amplitude",normalized_correlation )
    print("delay between signals", round(corr_delay), 'data points')    
  return normalized_correlation


def measure_euclidean(indoors, outdoors):
  norm_signal_indoors = indoors/ np.linalg.norm(indoors)
  norm_signal_outdoors = outdoors/ np.linalg.norm(outdoors)
  return distance.euclidean(norm_signal_indoors, norm_signal_outdoors)  

def signal_similarity(metadata,data_lookup,  zero_crossing_lookup, save_dir, verbose=False):
  similarity_data =[]
  for column_to_graph in COLUMNS_TO_GRAPH:
    indoors, outdoors=get_signal_indoors_outdoors(data_lookup, metadata, zero_crossing_lookup, column_to_graph)
    indoors_normrange, outdoors_normrange=normalize_signal_by_range(indoors.copy(), outdoors.copy())
    correlation_normrange = measure_correlation(indoors_normrange, outdoors_normrange, verbose=False)
    euc = measure_euclidean(indoors.copy(), outdoors.copy())
    cos = np.dot(indoors,outdoors)/(np.linalg.norm(indoors)*np.linalg.norm(outdoors))
    if verbose:
      print('\n',column_to_graph, COLUMNS_TO_AREA[column_to_graph],'\n','indoors vs outdoors ', "\nCosine Similarity:" , cos)
      print("Euclidean Distance:", euc)  
    similarity_data.append([column_to_graph, COLUMNS_TO_AREA[column_to_graph],cos, euc, correlation_normrange])
  sim_df = pd.DataFrame(similarity_data, columns=['column_to_graph', 'area', 'cosine_similarity', 'euclidean_distance', 'correlation_peak_divided_by_auto-correlation_amplitude'])
  sim_df.to_csv(os.path.join(save_dir, "signal_similarity.csv"))


def signal_sim_comb_legs(combined_legs,SAVE_DIR):
  save_dir = os.path.join(SAVE_DIR, 'combined_subjects_and_trials_and_legs')
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)  
  columns=['sensor', 'cosine_similarity', 'euclidean_distance', 'correlation_peak_divided_by_auto-correlation_amplitude']
  leg_similarities = []
  for sensors in list(combined_legs.keys()):
    indoors = combined_legs[sensors]['indoors']['avg']
    outdoors = combined_legs[sensors]['outdoors']['avg']
    indoors_normrange, outdoors_normrange = normalize_signal_by_range(indoors, outdoors)
    correlation_normrange= measure_correlation(indoors_normrange, outdoors_normrange, verbose=False)
    euc = measure_euclidean(indoors, outdoors)
    cos = np.dot(indoors,outdoors)/(np.linalg.norm(indoors)*np.linalg.norm(outdoors))
    leg_similarities.append([sensors, cos, euc, correlation_normrange])
  sim_df = pd.DataFrame(leg_similarities, columns=columns)
  sim_df.to_csv(os.path.join(save_dir, "signal_similarity_combined_legs.csv"))  

def signal_similarity_per_subject_indoor_outdoor(metadata,data_lookup, zero_crossing_lookup, save_dir):
  columns=['sensor','subjectID','area', 'cosine_similarity', 'euclidean_distance', 'correlation_peak_divided_by_auto-correlation_amplitude']
  data = []
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  for sensor in COLUMNS_TO_GRAPH:
    for subjectID in  metadata['subjectID'].unique():
      all_gates = aggregate_single_subject(data_lookup, metadata, zero_crossing_lookup, sensor, 'indoors' , subjectID)
      indoors = all_gates.mean(axis=0)
      all_gates = aggregate_single_subject(data_lookup, metadata, zero_crossing_lookup, sensor, 'outdoors' , subjectID)
      outdoors = all_gates.mean(axis=0)
      indoors_normrange, outdoors_normrange=normalize_signal_by_range(indoors.copy(), outdoors.copy())
      correlation_normrange = measure_correlation(indoors_normrange, outdoors_normrange, verbose=False)
      euc = measure_euclidean(indoors.copy(), outdoors.copy())
      cos = np.dot(indoors,outdoors)/(np.linalg.norm(indoors)*np.linalg.norm(outdoors))  
      data.append({'sensor':sensor,'subjectID':subjectID,'area':COLUMNS_TO_AREA[sensor],
                  'cosine_similarity':cos, 'euclidean_distance':euc, 
                  'correlation_peak_divided_by_auto-correlation_amplitude':correlation_normrange})    
  df_sp = pd.DataFrame(data, columns=columns)
  df_sp.to_csv(os.path.join(save_dir,'indoor_vs_outdoor.csv'))


def signal_similarity_per_subject_combined_invsout(metadata,data_lookup, zero_crossing_lookup, save_dir):
  '''compare the similarity between indoor and outdoor of the legs combined'''
  columns=['sensor','subjectID','area','cosine_similarity', 'euclidean_distance', 'correlation_peak_divided_by_auto-correlation_amplitude']
  data = []
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  for subjectID in  metadata['subjectID'].unique():
    for sensor_cols in COLUMNS_BY_SENSOR:
      sensor_name = sensor_cols['left'].replace(".1",' '+sensor_cols['sensor']).replace(".3",' '+sensor_cols['sensor'])
      indoors, outdoors = combine_legs_single_subject(sensor_name,data_lookup, metadata, zero_crossing_lookup, sensor_cols, subjectID )
      indoors_normrange, outdoors_normrange=normalize_signal_by_range(indoors.copy(), outdoors.copy())
      correlation_normrange = measure_correlation(indoors_normrange, outdoors_normrange, verbose=False)
      euc = measure_euclidean(indoors.copy(), outdoors.copy())
      cos = np.dot(indoors,outdoors)/(np.linalg.norm(indoors)*np.linalg.norm(outdoors))  
      data.append({'sensor':sensor_name,'subjectID':subjectID,'area':sensor_cols['sensor'],
                  'cosine_similarity':cos, 'euclidean_distance':euc, 
                  'correlation_peak_divided_by_auto-correlation_amplitude':correlation_normrange})    

  df_lrc = pd.DataFrame(data, columns=columns)
  df_lrc.to_csv(os.path.join(save_dir,'lr_combined_indoor_vs_outdoor.csv'))


def combine_legs_single_subject(sensor_name,data_lookup, metadata, zero_crossing_lookup, sensor_cols, subjectID ):
  left_gates_i  =aggregate_single_subject(data_lookup, metadata, zero_crossing_lookup, sensor_cols['left'], 'indoors' , subjectID)
  right_gates_i =aggregate_single_subject(data_lookup, metadata, zero_crossing_lookup, sensor_cols['right'], 'indoors' , subjectID)
  right_avg_i, aggg_gates_i = combine_legs_flip(left_gates_i, right_gates_i, sensor_name) 
  indoors = aggg_gates_i.mean(axis=0)
  left_gates_o  =aggregate_single_subject(data_lookup, metadata, zero_crossing_lookup, sensor_cols['left'], 'outdoors' , subjectID)
  right_gates_o =aggregate_single_subject(data_lookup, metadata, zero_crossing_lookup, sensor_cols['right'], 'outdoors' , subjectID)
  right_avg_o, aggg_gates_o = combine_legs_flip(left_gates_o, right_gates_o, sensor_name) 
  outdoors = aggg_gates_o.mean(axis=0)
  #plt.plot(right_avg)
  #plt.plot(aggg_gates.mean(axis=0))
  #plt.plot(left_gates_a.mean(axis=0)) 
  return indoors, outdoors


def signal_similarity_per_subject_left_right(metadata,data_lookup, zero_crossing_lookup, save_dir):
  '''compare the similarity of the mean signals from the left leg to the 
  right leg for each sensor and for indoor and outdoor '''
  columns=['sensor','subjectID','area','inout', 'cosine_similarity', 'euclidean_distance', 'correlation_peak_divided_by_auto-correlation_amplitude']
  data = []
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  for subjectID in  metadata['subjectID'].unique():
    for sensor_cols in COLUMNS_BY_SENSOR:
      sensor_name = sensor_cols['left'].replace(".1",' '+sensor_cols['sensor']).replace(".3",' '+sensor_cols['sensor'])
      for i, inout in enumerate(['indoors', 'outdoors']):
        left_gates_a  =aggregate_single_subject(data_lookup, metadata, zero_crossing_lookup, sensor_cols['left'], inout , subjectID)
        right_gates_a =aggregate_single_subject(data_lookup, metadata, zero_crossing_lookup, sensor_cols['right'], inout , subjectID)
      
        left_gates = left_gates_a.mean(axis=0)
        right_gates = right_gates_a.mean(axis=0)
        left_normrange, right_normrange=normalize_signal_by_range(left_gates.copy(), right_gates.copy())
        correlation_normrange = measure_correlation(left_normrange, right_normrange, verbose=False)
        euc = measure_euclidean(left_gates.copy(), right_gates.copy())
        cos = np.dot(left_gates,right_gates)/(np.linalg.norm(left_gates)*np.linalg.norm(right_gates))  

        data.append({'sensor':sensor_name,'subjectID':subjectID,'area':sensor_cols['sensor'],
                     'inout':inout,
                    'cosine_similarity':cos, 'euclidean_distance':euc, 
                    'correlation_peak_divided_by_auto-correlation_amplitude':correlation_normrange}) 
  df_sp = pd.DataFrame(data, columns=columns)
  df_sp.to_csv(os.path.join(save_dir,'left_vs_right.csv'))

def add_control_lr(sub,row , per_subject_data):
  '''subroutine for lr_control_ivo'''
  for sim_metric in ['cosine_similarity', 'euclidean_distance']:
    vals = sub[sim_metric].to_numpy()
    per_subject_data['indoor_'+sim_metric]=vals
    w_statistic, p_value = shapiro(vals)
    row.update({'indoor_'+sim_metric +'_avg':vals.mean(),
                'indoor_'+sim_metric +'_std': vals.std(),
                'indoor_'+sim_metric +'_p_shapiro':p_value,
                'indoor_'+sim_metric +'_w_shapiro':w_statistic})
    #print('metric', sim_metric, vals.mean(), '+-', vals.std())
    #print(sub.shape)


def add_indoor_vs_outdoor_each_side(row, io_sensors, df_io, per_subject_data):
  assert len(io_sensors)==2, "expecting one col for each leg"
  for ios in io_sensors:
    area = COLUMNS_TO_AREA[ios]
    side = area.replace("shank", "").replace("thigh", '').replace(" ", '')
    for sim_metric in ['cosine_similarity', 'euclidean_distance']:
      io_vals = df_io[df_io.sensor==ios][sim_metric].to_numpy()
      #print(len(io_vals))
      per_subject_data[side+'_i.vs.o_'+sim_metric]=io_vals
      w_statistic, p_value = shapiro(io_vals)
      row.update({side+'_i.vs.o_'+sim_metric +'_avg':io_vals.mean(),
                  side+'_i.vs.o_'+sim_metric +'_std': io_vals.std(),
                  side+'_i.vs.o_'+sim_metric +'_p_shapiro':p_value,
                  side+'_i.vs.o_'+sim_metric +'_w_shapiro':w_statistic,})


def add_ttest_wilcoxon(row, per_subject_data, p_alpha):
  row_cols = list(row.keys())
  test_type='t-test'
  for rcol in row_cols:
    if ('p_shapiro' in rcol) and (row[rcol]<p_alpha):
      test_type='wilcoxon'
  alternative = 'two-sided'
  for sim_metric in ['cosine_similarity', 'euclidean_distance']:
    control_key='indoor_'+sim_metric
    control_data = per_subject_data[control_key]
    per_subject_data.pop(control_key)
    compare_data_keys = [x for x in list(per_subject_data.keys()) if sim_metric in x]
    for compare_key in compare_data_keys:
      compare_data = per_subject_data[compare_key]
      if test_type=='t-test':
        t_statistic, p_value = ttest_rel(control_data, compare_data,alternative=alternative )
        row.update({compare_key+"_test_statistic":t_statistic, compare_key+"_test_p_value":p_value})          
      elif test_type=='wilcoxon':
        zero_method = 'wilcox'
        w_statistic, p_value = wilcoxon(control_data, compare_data,alternative=alternative,  zero_method= zero_method)
        row.update({compare_key+"_test_statistic":w_statistic, compare_key+"_test_p_value":p_value})          
  row['test_type'] = test_type



def map_sensor_lrc_to_sensor_per_leg(combined_leg_sensors, seperate_leg_sensors):
  sensor_to_lr = defaultdict(list)  
  for sensor in combined_leg_sensors:
    #print(sensor)
    leg_area=sensor.split(' ')[-1]
    sensor_beg = sensor.split('(')[0]
    for ios in seperate_leg_sensors:
      if sensor_beg in ios and leg_area in COLUMNS_TO_AREA[ios]:
        #print(ios)
        sensor_to_lr[sensor].append(ios)
        #print(COLUMNS_TO_AREA[ios])  
  return sensor_to_lr


def add_outdoor_lr(row, df_lr_o, sensor, per_subject_data):
  for sim_metric in ['cosine_similarity', 'euclidean_distance']:
    vals = df_lr_o[df_lr_o.sensor==sensor][sim_metric].to_numpy()
    w_statistic, p_value = shapiro(vals)
    per_subject_data['outdoor_'+sim_metric]=vals
    row.update({'outdoor_'+sim_metric +'_p_shapiro':p_value,'outdoor_'+sim_metric +'_w_shapiro':w_statistic })
    row.update({'outdoor_'+sim_metric +'_avg':vals.mean(),'outdoor_'+sim_metric +'_std': vals.std()})


def add_lrc_outdoor_vs_indoor(row,df_lrc, sensor, per_subject_data):
  for sim_metric in ['cosine_similarity', 'euclidean_distance']:
    vals = df_lrc[df_lrc.sensor==sensor][sim_metric].to_numpy()
    w_statistic, p_value = shapiro(vals)
    per_subject_data['lrc_i.vs.o_'+sim_metric]=vals
    row.update({'lrc_i.vs.o_'+sim_metric +'_p_shapiro':p_value,'lrc_i.vs.o_'+sim_metric +'_w_shapiro':w_statistic })
    row.update({'lrc_i.vs.o_'+sim_metric +'_avg':vals.mean(),'lrc_i.vs.o_'+sim_metric +'_std': vals.std()})

def add_z_scores_cohensd(row):
  for sim_metric in ['cosine_similarity', 'euclidean_distance']:
    control_in_key = 'indoor_'+sim_metric 
    avg = row[control_in_key+'_avg']
    std = row[control_in_key+'_std']
    for study_group in ['lrc_i.vs.o_', 'left_i.vs.o_', 'right_i.vs.o_', 'outdoor_']:
      X = row[study_group+sim_metric+'_avg']
      row[study_group+sim_metric+"_z_score"] = (X-avg)/std
      #print("control", control_in_key)
      #print(study_group+sim_metric+"_z_score")
      #print("X", X, "avg", avg, "std", std)
      pop_std = row[study_group+sim_metric+'_std']
      pooled_std = np.sqrt((std**2 + pop_std**2)/2)
      row[study_group+sim_metric+"_cohens_d"] = (X-avg)/pooled_std


def lr_control_ivo(df_lr, df_io, df_lrc, save_dir, p_alpha=0.05):
  '''lr is left right.
  ivo is indoor vs outdoor
  lrc is left leg and right leg data averaged together
  control is left vs right similarity metrics for indoors
  looks up the similarity metrics for each of the populations
  (i.e. outdoor left vs right) for each subject (30 data points)
  then takes mean and std and compares with the control population
  '''
  df_lr_o = df_lr[df_lr.inout=='outdoors']

  df_control = df_lr[df_lr.inout=='indoors']
  io_sensors = list(df_io.sensor.unique())  
  ## a list of sensor names without left or right leg specified, mapped to 
  ##    the names specifying right and left legs
  sensor_to_lr = map_sensor_lrc_to_sensor_per_leg(list(df_control.sensor.unique()),io_sensors )
  data=[]
  for sensor  in list(sensor_to_lr.keys()):
    sub = df_control[df_control.sensor==sensor]
    #print(sensor)
    row={}
    per_subject_data = {}
    io_sensors = sensor_to_lr[sensor]
    row['sensor']=sensor
    sensor = sub.sensor.iloc[0]
    add_control_lr(sub,row , per_subject_data)
    add_indoor_vs_outdoor_each_side(row, io_sensors, df_io, per_subject_data)
    add_outdoor_lr(row, df_lr_o, sensor, per_subject_data)
    add_lrc_outdoor_vs_indoor(row, df_lrc, sensor, per_subject_data)
    add_ttest_wilcoxon(row, per_subject_data, p_alpha)
    add_z_scores_cohensd(row)
    data.append(row)

  measurements = ['_avg', '_std','_p_shapiro','_w_shapiro']
  df_control_comp = pd.DataFrame(data, columns=list(row.keys()))
  df_control_comp.to_csv(os.path.join(save_dir, "lr_control_compare_lio_rio.csv"))
########################################################################################
########################################################################################

def compare_runs(base_dir, compare_dir):
  #base_files = glob.glob(os.path.join(base_dir, "trends_across_pace", "*.csv"))
  #compare_files = glob.glob(os.path.join(compare_dir, "trends_across_pace", "*.csv"))
  files_to_check = {'cos_euclidean_sim_stats.csv':'sensor','gate_swing_sim_stats.csv':'sensor',
                    'paired_comparison_sim_stats.csv':'sensor','peak_valley_sim_stats.csv':'sensor',
                    'poincare_sim_stats.csv':'sensor'}
  for fname in list(files_to_check.keys()):
    print("comparing", fname)
    base_df = pd.read_csv(os.path.join(base_dir,"trends_across_pace",fname), index_col=False)
    compare_df = pd.read_csv(os.path.join(compare_dir,"trends_across_pace",fname), index_col=False)
    base_df.sort_values(by=files_to_check[fname], inplace=True)
    compare_df.sort_values(by=files_to_check[fname], inplace=True)
    base_df.reset_index(drop=True, inplace=True)
    compare_df.reset_index(drop=True, inplace=True)
    print(base_df.compare(compare_df))

########################################################################################
########################################################################################

def run_everything1(metadata,data_lookup,  zero_crossing_lookup, SAVE_DIR):
  print("run_everything1", SAVE_DIR)
  save_dir = os.path.join(SAVE_DIR, "graph_each_subject_each_sensor")
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  each_sensor_each_subject(save_dir, data_lookup, metadata, zero_crossing_lookup)
  start = datetime.datetime.now()
  save_each_subject_each_sensor(save_dir, data_lookup, metadata, zero_crossing_lookup)
  print("time: save_each_subject_each_sensor", (datetime.datetime.now()-start).total_seconds(), "to run")
  save_dir = os.path.join(SAVE_DIR, 'poincare')
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  start = datetime.datetime.now()
  sens = LEFT_AVY_HEADER
  graph_poincare_per_leg(data_lookup, metadata, zero_crossing_lookup, sens, save_dir)
  print("time: graph_poincare_per_leg ", (datetime.datetime.now()-start).total_seconds(), "to run")
  combined_legs= aggregate_subjects_trials_legs(data_lookup, metadata, zero_crossing_lookup)
  for sensor in ['Acceleration Y (m/s^2) shank', 'Angular Velocity Y (rad/s) shank']:
    graph_poincare_comb_leg_per_sensor(combined_legs, sensor, save_dir, xlim=[-7,4], ylim=[-7,4])
  save_dir = os.path.join(SAVE_DIR, 'poincare',"per_subject")
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  plt.close()
  start = datetime.datetime.now()
  df_poincare_p_subject = graph_poincare_comb_leg_per_subject(metadata,data_lookup,zero_crossing_lookup, save_dir)
  plt.close()
  print("time: graph_poincare_comb_leg_per_subject", (datetime.datetime.now()-start).total_seconds(), "to run")
  poincare_sim_stats_per_sensor(SAVE_DIR)
  print("time: poincare_sim_stats_per_sensor", (datetime.datetime.now()-start).total_seconds(), "to run")

def run_everything2(metadata,data_lookup,  zero_crossing_lookup, SAVE_DIR):
  print("run_everything2", SAVE_DIR)  
  start = datetime.datetime.now()
  gate_peak_valley_swing(metadata, data_lookup, zero_crossing_lookup, SAVE_DIR)
  print("time: gate_peak_valley_swing", (datetime.datetime.now()-start).total_seconds(), "to run")
  start = datetime.datetime.now()
  calc_shapiro_t_test(SAVE_DIR)
  print("time: calc_shapiro_t_test ", (datetime.datetime.now()-start).total_seconds(), "to run")
  plt.close()
  start = datetime.datetime.now()
  calc_shapiro_t_test_legs_combined(SAVE_DIR)
  print("time: calc_shapiro_t_test_legs_combined", (datetime.datetime.now()-start).total_seconds(), "to run")
  start = datetime.datetime.now()
  save_dir = os.path.join(SAVE_DIR, 'combined_subjects_and_trials')
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  graph_sensors_combined_subjects_trials(save_dir, data_lookup, metadata, zero_crossing_lookup)
  plt.close()
  print("time: graph_sensors_combined_subjects_trials", (datetime.datetime.now()-start).total_seconds(), "to run")
  start = datetime.datetime.now()
  global_mins, global_maxes, ranges = combined_subjects_trials_signal_stats(data_lookup, metadata, zero_crossing_lookup, save_dir)
  print("time: combined_subjects_trials_signal_stats", (datetime.datetime.now()-start).total_seconds(), "to run")
  plt.close()
  start = datetime.datetime.now()
  save_dir = os.path.join(SAVE_DIR, 'combined_subjects_and_trials_and_legs')
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)   
  combined_legs = graph_aggregate_subjects_trials_legs(save_dir, data_lookup, metadata, zero_crossing_lookup)
  plt.close()
  print("time: graph_aggregate_subjects_trials_legs", (datetime.datetime.now()-start).total_seconds(), "to run")
  return combined_legs

def run_everything3(metadata,data_lookup,  zero_crossing_lookup, SAVE_DIR, combined_legs):
  print("run_everything3", SAVE_DIR)
  start = datetime.datetime.now()
  signal_similarity(metadata,data_lookup, zero_crossing_lookup, os.path.join(SAVE_DIR, 'combined_subjects_and_trials'))
  print("time: signal_similarity", (datetime.datetime.now()-start).total_seconds(), "to run")
  start = datetime.datetime.now()
  signal_similarity_per_subject_indoor_outdoor(metadata,data_lookup, zero_crossing_lookup, os.path.join(SAVE_DIR, 'similarity_per_subject'))
  print("time: signal_similarity_per_subject_indoor_outdoor", (datetime.datetime.now()-start).total_seconds(), "to run")
  plt.close()
  start = datetime.datetime.now()
  save_dir=os.path.join(SAVE_DIR, 'similarity_per_subject')
  signal_similarity_per_subject_combined_invsout(metadata,data_lookup, zero_crossing_lookup, save_dir)
  print("time: signal_similarity_per_subject_combined_invsout", (datetime.datetime.now()-start).total_seconds(), "to run")
  start = datetime.datetime.now()
  signal_similarity_per_subject_left_right(metadata,data_lookup, zero_crossing_lookup, os.path.join(SAVE_DIR, 'similarity_per_subject'))
  print("time: signal_similarity_per_subject_left_right", (datetime.datetime.now()-start).total_seconds(), "to run")
  start = datetime.datetime.now()
  save_dir=os.path.join(SAVE_DIR, 'similarity_per_subject')
  df_lr=pd.read_csv(os.path.join(save_dir,'left_vs_right.csv'))
  df_io=pd.read_csv(os.path.join(save_dir,'indoor_vs_outdoor.csv'))
  df_lrc_io=pd.read_csv(os.path.join(save_dir,'lr_combined_indoor_vs_outdoor.csv'))

  lr_control_ivo(df_lr, df_io,df_lrc_io, save_dir)
  print("time: lr_control_ivo", (datetime.datetime.now()-start).total_seconds(), "to run")
  start = datetime.datetime.now()
  #combined_legs = aggregate_subjects_trials_legs(data_lookup, metadata, zero_crossing_lookup)
  signal_sim_comb_legs(combined_legs, SAVE_DIR)
  plt.close()
  print("time: signal_sim_comb_legs", (datetime.datetime.now()-start).total_seconds(), "to run")

def run_cadence_filtered_everything():
  ##load all data and filter it
  this_file_path = os.path.dirname(os.path.realpath(__file__))
  base_dir = os.path.join(this_file_path,'results','05.15.23')
  outliers = cadence_remove_outlier(base_dir)
  if not os.path.join(base_dir, 'cadence_filtered'):
    os.mkdir(os.path.join(base_dir, 'cadence_filtered'))
  for PACE in ['slow', 'normal','fast']:
    #PACE = 'normal'
    load_dir = os.path.join(base_dir,PACE)
    SAVE_DIR = os.path.join(base_dir,'cadence_filtered',PACE)
    if not os.path.exists(SAVE_DIR):
      os.mkdir(SAVE_DIR)
    print("SAVE_DIR", SAVE_DIR)
    load_dir = os.path.join(load_dir,'pickles')
    with open(os.path.join(load_dir,"metadata.pickle"), 'rb') as fileobj:
      metadata = pickle.load(fileobj)   
    for subj in outliers:
      metadata = metadata[metadata['subjectID']!=subj]
    start = datetime.datetime.now()
    data_lookup = {}
    for filename in metadata['filename']:
      data_lookup[filename]=load_data(filename)
    print("time: data_lookup", (datetime.datetime.now()-start).total_seconds(), "to run")  
    start = datetime.datetime.now()
    ##doesn't save anything
    zero_crossing_lookup =calc_all_gate_crossings(metadata, data_lookup, GATE_CROSSING)
    print("time:  zero_crossing_lookup", (datetime.datetime.now()-start), "to run")  
    save_dir_gate_lengths = os.path.join(SAVE_DIR, "stats_of_gate_lengths")
    if not os.path.exists(save_dir_gate_lengths):
      os.mkdir(save_dir_gate_lengths)
    print("sub_dirs", os.listdir(SAVE_DIR))
    start = datetime.datetime.now()
    df_gate_stats_cols = ['sensor','area', 'in-out', 'filename','trial', 'subjectID' ,'avg gate length (data points)', 'std', 'max', 'min', 'data points per file', 'vertical_gate_crossing' ]
    ##saves data
    df_per_file, filter_to_gate_thresh = stats_gate_lengths_by_file(metadata,data_lookup, df_gate_stats_cols, save_dir_gate_lengths, MAX_STD= 2, zero_crossing_lookup=zero_crossing_lookup)
    print("time: filter_to_gate_thresh", (datetime.datetime.now()-start).total_seconds(), "to run")
    start = datetime.datetime.now()
    zero_crossing_lookup=calc_all_gate_crossings(metadata, data_lookup, gate_crossing = GATE_CROSSING, gate_length_bounds= filter_to_gate_thresh)
    ##re-calculate the stats of the gate lengths now that the filter has been applied
    df_gate_stats_cols = ['sensor','area', 'in-out', 'filename','trial', 'subjectID' ,'avg gate length (data points)', 'std', 'max', 'min', 'data points per file', 'vertical_gate_crossing' ]
    df_per_file_filtered, _ = stats_gate_lengths_by_file(metadata,data_lookup, df_gate_stats_cols, save_dir_gate_lengths,
                                                        fname_gate_length_file = "per_file_2std_filtered_outliers" , MAX_STD= 2, zero_crossing_lookup=zero_crossing_lookup)
    print("re-calculate the zero crossings with filter applied")
    print("took", (datetime.datetime.now()-start), "to run")
    run_everything1(metadata,data_lookup,  zero_crossing_lookup, SAVE_DIR)
    combined_legs= run_everything2(metadata,data_lookup,  zero_crossing_lookup, SAVE_DIR)
    plt.close()
    run_everything3(metadata,data_lookup,  zero_crossing_lookup, SAVE_DIR, combined_legs)
    df_cadence=cadence_per_subject(SAVE_DIR,metadata, zero_crossing_lookup)
    print("saving pickles")
    save_dir = os.path.join(SAVE_DIR,'pickles')
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)
    with open(os.path.join(save_dir,"metadata.pickle"), 'wb') as fileobj:
      pickle.dump(metadata,fileobj)
    with open(os.path.join(save_dir,"data_lookup.pickle"), 'wb') as fileobj:
      pickle.dump(data_lookup,fileobj)
    with open(os.path.join(save_dir,"zero_crossing_lookup.pickle"), 'wb') as fileobj:
      pickle.dump(zero_crossing_lookup,fileobj)      

def run_everything():
  print("if your computer goes to sleep while this is running, the function will hang and never finish")
  function_start = datetime.datetime.now()
  print("running", datetime.datetime.today().strftime('%Y-%m-%d'))
  ##load all data and filter it
  this_file_path = os.path.dirname(os.path.realpath(__file__))
  base_dir = os.path.join(this_file_path,'results','05.17.23')
  if not os.path.exists(base_dir):
    os.mkdir(base_dir)
  for PACE in [ 'fast', 'normal','slow']:
    pace_start = datetime.datetime.now()
    SAVE_DIR = os.path.join(base_dir,PACE)
    print("SAVE_DIR", SAVE_DIR)
    if not os.path.exists(SAVE_DIR):
      os.mkdir(SAVE_DIR)
    start = datetime.datetime.now()
    metadata=load_metadata(PACE, DATA_DIR)
    print("paces found", metadata['pace'].unique())
    print("time: load_metadata", (datetime.datetime.now()-start).total_seconds(), "to run")  
    assert len(metadata['pace'].unique())==1, "paces"+str(metadata['pace'].unique())+" rows "+str(len(metadata))
    start = datetime.datetime.now()
    data_lookup = {}
    for filename in metadata['filename']:
      data_lookup[filename]=load_data(filename)
    print("time: data_lookup", (datetime.datetime.now()-start).total_seconds(), "to run")  
    start = datetime.datetime.now()
    ##doesn't save anything
    zero_crossing_lookup =calc_all_gate_crossings(metadata, data_lookup, gate_crossing = GATE_CROSSING)
    print("time: zero_crossing_lookup", (datetime.datetime.now()-start), "to run")  
    save_dir_gate_lengths = os.path.join(SAVE_DIR, "stats_of_gate_lengths")
    if not os.path.exists(save_dir_gate_lengths):
      os.mkdir(save_dir_gate_lengths)
    print("sub_dirs", os.listdir(SAVE_DIR))

    start = datetime.datetime.now()
    df_gate_stats_cols = ['sensor','area', 'in-out', 'filename','trial', 'subjectID' ,'avg gate length (data points)', 'std', 'max', 'min', 'data points per file', 'vertical_gate_crossing' ]
    ##saves data
    df_per_file, filter_to_gate_thresh = stats_gate_lengths_by_file(metadata,data_lookup, df_gate_stats_cols, save_dir_gate_lengths, MAX_STD= 2, zero_crossing_lookup=zero_crossing_lookup)
    print("time: filtering bad gates", (datetime.datetime.now()-start).total_seconds(), "to run")
    start = datetime.datetime.now()
    zero_crossing_lookup=calc_all_gate_crossings(metadata, data_lookup, gate_crossing = GATE_CROSSING, gate_length_bounds= filter_to_gate_thresh)
    ##re-calculate the stats of the gate lengths now that the filter has been applied
    df_gate_stats_cols = ['sensor','area', 'in-out', 'filename','trial', 'subjectID' ,'avg gate length (data points)', 'std', 'max', 'min', 'data points per file', 'vertical_gate_crossing' ]
    df_per_file_filtered, _ = stats_gate_lengths_by_file(metadata,data_lookup, df_gate_stats_cols, save_dir_gate_lengths,
                                                        fname_gate_length_file = "per_file_2std_filtered_outliers" , MAX_STD= 2, zero_crossing_lookup=zero_crossing_lookup)
    print("re-calculate the zero crossings with filter applied")
    print("took", (datetime.datetime.now()-start), "to run")
    run_everything1(metadata,data_lookup,  zero_crossing_lookup, SAVE_DIR)
    combined_legs= run_everything2(metadata,data_lookup,  zero_crossing_lookup, SAVE_DIR)
    plt.close()
    run_everything3(metadata,data_lookup,  zero_crossing_lookup, SAVE_DIR, combined_legs)
    df_cadence=cadence_per_subject(SAVE_DIR,metadata, zero_crossing_lookup)
    print("saving pickles")
    save_dir = os.path.join(SAVE_DIR,'pickles')
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)
    with open(os.path.join(save_dir,"metadata.pickle"), 'wb') as fileobj:
      pickle.dump(metadata,fileobj)
    with open(os.path.join(save_dir,"data_lookup.pickle"), 'wb') as fileobj:
      pickle.dump(data_lookup,fileobj)
    with open(os.path.join(save_dir,"zero_crossing_lookup.pickle"), 'wb') as fileobj:
      pickle.dump(zero_crossing_lookup,fileobj)     
    SAVE_DIR = None
    zero_crossing_lookup=None
    data_lookup=None
    metadata=None
    print(PACE, "finished", (datetime.datetime.now()-pace_start))

  if not os.path.exists(os.path.join(base_dir,'trends_across_pace')):
    os.mkdir(os.path.join(base_dir,'trends_across_pace'))
  re_formatting_peak(base_dir)
  re_format_paired_comparison(base_dir)
  re_format_distance_sim(base_dir)
  re_format_poincare_sim_stats(base_dir)
  print("time: run_everything",(datetime.datetime.now()-function_start) )

if __name__=="__main__":
  #run_everything()
  compare_runs(r".\results\05_13_23", r".\results\05.17.23")












