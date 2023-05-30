import os
import glob
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple
from load_data import extract_trial_data
from collections import Counter
from global_variables import RIGHT_AVY_HEADER,LEFT_AVY_HEADER,  FREQUENCY, GATE_CROSSING, DATA_DIR, COLUMNS_BY_SENSOR, COLUMNS_TO_LEG, COLUMNS_TO_AREA, COLUMNS_TO_GRAPH
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