import os, datetime, pickle
import numpy as np
import matplotlib.pyplot as plt
from gate_crossings import calc_all_gate_crossings, stats_gate_lengths_by_file
from aggregating_gates import each_sensor_each_subject, save_each_subject_each_sensor, aggregate_subjects_trials_legs, graph_aggregate_subjects_trials_legs
from aggregating_gates import gate_peak_valley_swing, calc_shapiro_t_test, calc_shapiro_t_test_legs_combined, graph_sensors_combined_subjects_trials
from aggregating_gates import combined_subjects_trials_signal_stats, cadence_per_subject
from poincare import graph_poincare_comb_leg_per_sensor, graph_poincare_per_leg, graph_poincare_comb_leg_per_subject, poincare_sim_stats_per_sensor
from globals import LEFT_AVY_HEADER, GATE_CROSSING, DATA_DIR
from aggregate_speeds import re_formatting_peak, re_format_paired_comparison, re_format_distance_sim, re_format_poincare_sim_stats
from load_data import load_data,  load_metadata
from similarity import signal_similarity, signal_similarity_per_subject_indoor_outdoor, signal_similarity_per_subject_combined_invsout
from similarity import signal_similarity_per_subject_left_right, signal_sim_comb_legs, lr_control_ivo
import pandas as pd

def run_everything1(metadata,data_lookup,  zero_crossing_lookup, SAVE_DIR):
  print("run_everything1", SAVE_DIR)
  save_dir = os.path.join(SAVE_DIR, "graph_each_subject_each_sensor")
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  each_sensor_each_subject(save_dir, data_lookup, metadata, zero_crossing_lookup)
  start = datetime.datetime.now()
  save_each_subject_each_sensor(save_dir, data_lookup, metadata, zero_crossing_lookup)
  print("save_each_subject_each_sensor took", (datetime.datetime.now()-start).total_seconds(), "to run")
  save_dir = os.path.join(SAVE_DIR, 'poincare')
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  start = datetime.datetime.now()
  sens = LEFT_AVY_HEADER
  graph_poincare_per_leg(data_lookup, metadata, zero_crossing_lookup, sens, save_dir)
  print("graph_poincare_per_leg took", (datetime.datetime.now()-start).total_seconds(), "to run")
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
  print("graph_poincare_comb_leg_per_subject took", (datetime.datetime.now()-start).total_seconds(), "to run")
  poincare_sim_stats_per_sensor(SAVE_DIR)
  print("poincare data took", (datetime.datetime.now()-start).total_seconds(), "to run")
def run_everything2(metadata,data_lookup,  zero_crossing_lookup, SAVE_DIR):
  print("run_everything2", SAVE_DIR)
  start = datetime.datetime.now()
  gate_peak_valley_swing(metadata, data_lookup, zero_crossing_lookup, SAVE_DIR)
  print("gate_peak_valley_swing took", (datetime.datetime.now()-start).total_seconds(), "to run")
  start = datetime.datetime.now()
  ##FLAG
  calc_shapiro_t_test(SAVE_DIR)
  print("calc_shapiro_t_test took", (datetime.datetime.now()-start).total_seconds(), "to run")
  plt.close()
  start = datetime.datetime.now()
  ##FLAG
  calc_shapiro_t_test_legs_combined(SAVE_DIR)
  print("calc_shapiro_t_test_legs_combined took", (datetime.datetime.now()-start).total_seconds(), "to run")
  start = datetime.datetime.now()
  save_dir = os.path.join(SAVE_DIR, 'combined_subjects_and_trials')
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  graph_sensors_combined_subjects_trials(save_dir, data_lookup, metadata, zero_crossing_lookup)
  plt.close()
  print("graph_sensors_combined_subjects_trials took", (datetime.datetime.now()-start).total_seconds(), "to run")
  start = datetime.datetime.now()
  global_mins, global_maxes, ranges = combined_subjects_trials_signal_stats(data_lookup, metadata, zero_crossing_lookup, save_dir)
  print("combined_subjects_trials_signal_statstook", (datetime.datetime.now()-start).total_seconds(), "to run")
  plt.close()
  start = datetime.datetime.now()
  save_dir = os.path.join(SAVE_DIR, 'combined_subjects_and_trials_and_legs')
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  combined_legs = graph_aggregate_subjects_trials_legs(save_dir, data_lookup, metadata, zero_crossing_lookup)
  plt.close()
  print("graph_aggregate_subjects_trials_legs took", (datetime.datetime.now()-start).total_seconds(), "to run")
  return combined_legs
def run_everything3(metadata,data_lookup,  zero_crossing_lookup, SAVE_DIR, combined_legs):
  print("run_everything3", SAVE_DIR)
  start = datetime.datetime.now()
  signal_similarity(metadata,data_lookup, zero_crossing_lookup, os.path.join(SAVE_DIR, 'combined_subjects_and_trials'))
  print("signal_similarity took", (datetime.datetime.now()-start).total_seconds(), "to run")
  start = datetime.datetime.now()
  signal_similarity_per_subject_indoor_outdoor(metadata,data_lookup, zero_crossing_lookup, os.path.join(SAVE_DIR, 'similarity_per_subject'))
  print("signal_similarity_per_subject_indoor_outdoor took", (datetime.datetime.now()-start).total_seconds(), "to run")
  plt.close()
  start = datetime.datetime.now()
  save_dir=os.path.join(SAVE_DIR, 'similarity_per_subject')
  signal_similarity_per_subject_combined_invsout(metadata,data_lookup, zero_crossing_lookup, save_dir)
  print("signal_similarity_per_subject_combined_invsout took", (datetime.datetime.now()-start).total_seconds(), "to run")
  start = datetime.datetime.now()
  signal_similarity_per_subject_left_right(metadata,data_lookup, zero_crossing_lookup, os.path.join(SAVE_DIR, 'similarity_per_subject'))
  print("signal_similarity_per_subject_left_right took", (datetime.datetime.now()-start).total_seconds(), "to run")
  start = datetime.datetime.now()
  save_dir=os.path.join(SAVE_DIR, 'similarity_per_subject')
  df_lr=pd.read_csv(os.path.join(save_dir,'left_vs_right.csv'))
  df_io=pd.read_csv(os.path.join(save_dir,'indoor_vs_outdoor.csv'))
  df_lrc_io=pd.read_csv(os.path.join(save_dir,'lr_combined_indoor_vs_outdoor.csv'))

  lr_control_ivo(df_lr, df_io,df_lrc_io, save_dir)
  print("lr_control_ivo took", (datetime.datetime.now()-start).total_seconds(), "to run")
  start = datetime.datetime.now()
  #combined_legs = aggregate_subjects_trials_legs(data_lookup, metadata, zero_crossing_lookup)
  signal_sim_comb_legs(combined_legs, SAVE_DIR)
  plt.close()
  print("signal_sim_comb_legs took", (datetime.datetime.now()-start).total_seconds(), "to run")

def run_cadence_filtered_everything():
  ##load all data and filter it
  with open(os.path.join("Analysis", 'cadence','outliers.pickle'),'rb') as fileobj:
    outliers = pickle.load(fileobj)
  for PACE in ['slow', 'normal','fast']:
    #PACE = 'normal'
    load_dir = os.path.join('Analysis',PACE)
    SAVE_DIR = os.path.join('Analysis','cadence_filtered',PACE)
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
    print("data_lookup took", (datetime.datetime.now()-start).total_seconds(), "to run")
    start = datetime.datetime.now()
    ##doesn't save anything
    zero_crossing_lookup =calc_all_gate_crossings(metadata, data_lookup, GATE_CROSSING)
    print("zero_crossing_lookup took", (datetime.datetime.now()-start), "to run")
    save_dir_gate_lengths = os.path.join(SAVE_DIR, "stats_of_gate_lengths")
    if not os.path.exists(save_dir_gate_lengths):
      os.mkdir(save_dir_gate_lengths)
    start = datetime.datetime.now()
    df_gate_stats_cols = ['sensor','area', 'in-out', 'filename','trial', 'subjectID' ,'avg gate length (data points)', 'std', 'max', 'min', 'data points per file', 'vertical_gate_crossing' ]
    ##saves data
    df_per_file, filter_to_gate_thresh = stats_gate_lengths_by_file(metadata,data_lookup, df_gate_stats_cols, save_dir_gate_lengths, MAX_STD= 2, zero_crossing_lookup=zero_crossing_lookup)
    print("filtering bad gates took", (datetime.datetime.now()-start).total_seconds(), "to run")
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
  ##load all data and filter it
  base_dir = os.path.join('Analysis',"05_13_23")
  if not os.path.exists(base_dir):
    os.mkdir(base_dir)
  for PACE in [ 'fast', 'normal','slow']:
    SAVE_DIR = os.path.join(base_dir,PACE)
    print("SAVE_DIR", SAVE_DIR)
    if not os.path.exists(SAVE_DIR):
      os.mkdir(SAVE_DIR)
    start = datetime.datetime.now()
    metadata=load_metadata(PACE, DATA_DIR)
    print("paces found", metadata['pace'].unique())
    print("metadata took", (datetime.datetime.now()-start).total_seconds(), "to run")

    start = datetime.datetime.now()
    data_lookup = {}
    for filename in metadata['filename']:
      data_lookup[filename]=load_data(filename)
    print("data_lookup took", (datetime.datetime.now()-start).total_seconds(), "to run")
    start = datetime.datetime.now()
    ##doesn't save anything
    zero_crossing_lookup =calc_all_gate_crossings(metadata, data_lookup, GATE_CROSSING)
    print("zero_crossing_lookup took", (datetime.datetime.now()-start), "to run")
    save_dir_gate_lengths = os.path.join(SAVE_DIR, "stats_of_gate_lengths")
    if not os.path.exists(save_dir_gate_lengths):
      os.mkdir(save_dir_gate_lengths)
    start = datetime.datetime.now()
    df_gate_stats_cols = ['sensor','area', 'in-out', 'filename','trial', 'subjectID' ,'avg gate length (data points)', 'std', 'max', 'min', 'data points per file', 'vertical_gate_crossing' ]
    ##saves data
    df_per_file, filter_to_gate_thresh = stats_gate_lengths_by_file(metadata,data_lookup, df_gate_stats_cols, save_dir_gate_lengths, MAX_STD= 2, zero_crossing_lookup=zero_crossing_lookup)
    print("filtering bad gates took", (datetime.datetime.now()-start).total_seconds(), "to run")
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
    if not os.path.exists(os.path.join(base_dir,'trends_across_pace')):
        os.mkdir(os.path.join(base_dir,'trends_across_pace'))
    re_formatting_peak(base_dir)
    re_format_paired_comparison(base_dir)
    re_format_distance_sim(base_dir)
    re_format_poincare_sim_stats(base_dir)

run_everything()