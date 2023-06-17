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

#from scipy.signal import resample
from scipy.signal import correlate, find_peaks, butter, sosfilt

from scipy.stats import shapiro, ttest_rel, wilcoxon

import pickle

from load_data import select_random_df, load_data, load_metadata
from extract_gates import find_swing_stance_index,  find_lowest_valley, avg_std_gate_lengths, max_peak, calc_all_gate_crossings, stats_gate_lengths_by_file
from poincare import poincare
from re_formatting_across_pace import peak_z_score_cohens_d, re_format_paired_comparison, re_format_distance_sim, re_format_poincare_sim_stats
from global_variables import RIGHT_AVY_HEADER,LEFT_AVY_HEADER,  FREQUENCY, GATE_CROSSING, DATA_DIR, COLUMNS_BY_SENSOR, COLUMNS_TO_LEG, COLUMNS_TO_AREA, COLUMNS_TO_GRAPH
from extract_signal import each_sensor_each_subject, save_each_subject_each_sensor, aggregate_subjects_trials_legs, combined_subjects_trials_signal_stats, graph_sensors_combined_subjects_trials
from extract_signal import calc_shapiro_t_test, graph_aggregate_subjects_trials_legs, cadence_remove_outlier, cadence_per_subject
from stp import gate_peak_valley_swing, calc_shapiro_t_test_legs_combined, re_formatting_peak
from extract_signal import aggregate_single_subject
from signal_similarity import signal_similarity, lr_control_ivo, signal_similarity_per_subject_indoor_outdoor, signal_similarity_per_subject_left_right, signal_similarity_per_subject_combined_invsout
from signal_similarity import signal_sim_comb_legs, combine_legs_single_subject
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
def graph_poincare_comb_leg(combined_legs, save_dir_m, metadata):
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
def calculate_poincare_stats(SAVE_DIR, alpha = 0.05):
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
  peak_z_score_cohens_d(base_dir)
  print("time: run_everything",(datetime.datetime.now()-function_start) )


if __name__=="__main__":
  run_everything()
  compare_runs(r".\results\05_13_23", r".\results\05.17.23")

