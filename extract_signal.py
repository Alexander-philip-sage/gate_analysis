from global_variables import RIGHT_AVY_HEADER,LEFT_AVY_HEADER,  FREQUENCY, GATE_CROSSING, DATA_DIR, COLUMNS_BY_SENSOR, COLUMNS_TO_LEG, COLUMNS_TO_AREA, COLUMNS_TO_GRAPH
import matplotlib.pyplot as plt
import os
import glob
import datetime
import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy.signal import correlate, find_peaks, butter, sosfilt
from scipy.interpolate import interp1d
from load_data import select_random_df
import random
from extract_gates import find_swing_stance_index, find_lowest_valley, max_peak, avg_std_gate_lengths
import pickle
from scipy.stats import shapiro, ttest_rel, wilcoxon

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

def graph_normalizing_affect(metadata, zero_crossing_lookup,data_lookup):
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
def graph_random_thighs(metadata,data_lookup, zero_crossing_lookup):
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