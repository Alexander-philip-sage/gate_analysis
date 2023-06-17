import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple
import numpy as np
from global_variables import RIGHT_AVY_HEADER,LEFT_AVY_HEADER,  FREQUENCY, GATE_CROSSING, DATA_DIR, COLUMNS_BY_SENSOR, COLUMNS_TO_LEG, COLUMNS_TO_AREA, COLUMNS_TO_GRAPH
import glob
from extract_gates import find_swing_stance_index, avg_std_gate_lengths
from extract_signal import calc_shapiro, aggregate_single_subject, get_zeros_crossings_single_subject, calculate_peaks_valley_range


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
    print(comb_file_path)
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
def gate_peak_valley_swing_save_data(columns_pkvgs, save_dir, sensor, metadata, data_lookup, zero_crossing_lookup ):
  assert len(metadata['pace'].unique())==1
  pace=metadata['pace'].unique()[0]
  df_filename = sensor.replace(r"/", '-')+'_'+pace
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


