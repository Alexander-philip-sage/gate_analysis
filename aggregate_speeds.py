import os, datetime, pickle
import numpy as np
import pandas as pd
import glob
from scipy.stats import shapiro, ttest_rel, wilcoxon
from poincare  import poincare_z_score_cohens_d, calc_t_test_poincare, calc_wilcoxon_poincare
from poincare import indoor_outdoor_similarity
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

def pull_together_across_speeds(save=True):
  '''pulls the cadence values together across speeds'''
  if save:
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
  if save:
    df_compare.to_csv(os.path.join(cadence_dir, "cadence_aggregate.csv"))
  return df_compare

def fluff_shape(pace1_data, pace2_data):
  '''artificially inflates the number of subjects so that
  all paces have the same number of data points'''
  while pace2_data.shape[0]<pace1_data.shape[0]:
    pace2_data = np.append(pace2_data, pace2_data.mean())
  while pace1_data.shape[0]<pace2_data.shape[0]:
    pace1_data = np.append(pace1_data, pace1_data.mean())
  #print("shape 1",  pace1_data.shape, "shape 2",  pace2_data.shape)
  return pace1_data, pace2_data

def pace_similarity(pace1, pace2, compare_cadence, inout, alpha):
  pace1_data = compare_cadence[pace1][inout]
  pace2_data= compare_cadence[pace2][inout]
  pace1_data, pace2_data= fluff_shape(pace2_data,pace1_data)
  #print("shape 1", pace1, pace1_data.shape, "shape 2", pace2, pace2_data.shape)
  z_score, cohens_d=poincare_z_score_cohens_d(pace1_data, pace2_data)
  shapiro_statistic_pace1_data, p_pace1_data = shapiro(pace1_data)
  shapiro_statistic_pace2_data, p_pace2_data = shapiro(pace2_data)
  if (p_pace1_data >alpha) and (p_pace2_data> alpha):
    test_stat, p_value=calc_t_test_poincare(pace1_data, pace2_data)
    test_type='t_test'
  else:
    test_stat, p_value=calc_wilcoxon_poincare(pace1_data, pace2_data)
    test_type='wilcoxon'
  row = {'label':inout+'_'+pace1+"_vs_"+pace2,'inout':inout, "pace1":pace1,
         'shapiro_stat_pace1':shapiro_statistic_pace1_data,
         'shapiro_p_val_pace1':p_pace1_data, "pace2":pace2,
         'shapiro_stat_pace2':shapiro_statistic_pace2_data,
         'shapiro_p_val_pace2':p_pace2_data,
         'test_type':test_type, 'stat':test_stat, 'p_value':p_value,
         'z_score':z_score, "cohens_d":cohens_d}
  return row

def cadence_similarity_across_speeds(save=True):
  '''pulls the cadence values together across speeds'''
  if save:
    cadence_dir =os.path.join("Analysis", "05_13_23", 'cadence')
    if not os.path.exists(cadence_dir):
      os.mkdir(cadence_dir)
  compare_cadence = {}
  indoor_vs_outdoor_data = []
  speed_compare = []
  for speed in ['slow', 'normal','fast']:
    lookup_dir = os.path.join("Analysis","05_13_23",speed, 'stats_of_gate_lengths')
    filename = glob.glob(os.path.join(lookup_dir,"per_subject_*.csv"))[0]
    print("looking at file", filename)
    df_cadence=pd.read_csv(filename)
    indoors = df_cadence[df_cadence['inout']=='indoors']['cadence_avg_step_p_minute'].values
    outdoors = df_cadence[df_cadence['inout']=='outdoors']['cadence_avg_step_p_minute'].values
    compare_cadence[speed] = {'indoors':indoors, 'outdoors':outdoors}
    indoor_row, outdoor_row = indoor_outdoor_similarity(indoors, outdoors, speed, alpha=0.05)
    indoor_vs_outdoor_data.extend([indoor_row, outdoor_row])
  for inout in ['indoors', 'outdoors']:
    row = pace_similarity('normal', 'slow', compare_cadence, inout,  alpha=0.05)
    speed_compare.append(row)
    row = pace_similarity('normal','fast', compare_cadence, inout,  alpha=0.05)
    speed_compare.append(row)
    row = pace_similarity('fast','slow', compare_cadence, inout,  alpha=0.05)
    speed_compare.append(row)
  df_compare = pd.DataFrame(speed_compare)
  df_ivso = pd.DataFrame(indoor_vs_outdoor_data)
  if save:
    df_ivso.to_csv(os.path.join(cadence_dir, "indoor_vs_outdoor_similarity.csv"))
    df_compare.to_csv(os.path.join(cadence_dir, "speed_similarity.csv"))
  return df_compare, df_ivso


def re_format_poincare_sim_stats(base_dir = os.path.join('Analysis')):
  df=pd.DataFrame()
  for speed in ['normal', 'slow', 'fast']:
    load_dir = os.path.join(base_dir, speed,'poincare', 'per_subject')
    df_p = pd.read_csv(os.path.join(load_dir, 'poincare_sim_stats_per_sensor.csv'))
    df_p['pace']=speed
    df = pd.concat([df, df_p])
  sensors = list(df['sensor'].unique())
  new_table = []
  test_type_table = []
  for sensor in sensors:
    subdf = df[(df['sensor']==sensor)&(df['inout']=='outdoors')]
    row = [sensor]
    test_type_row = [sensor]
    column_names = ['sensor']
    test_type_names = ['sensor']
    for speed in ['normal', 'slow', 'fast']:
      for sd in ['sd1', 'sd2']:
        test_type_row.append(subdf[(subdf['source']==sd)&(subdf['pace']==speed)]['test_type'].iloc[0])
        test_type_names.append(speed +' '+sd)
        for metric in ['p_value', 'z_score']:
          column_names.append(speed +' '+sd+' '+metric)
          cond = (subdf['source']==sd)&(subdf['pace']==speed)
          val=subdf[cond][metric].iloc[0]
          row.append(val)
    new_table.append(row)
    test_type_table.append(test_type_row)
  new_table= pd.DataFrame(new_table, columns=column_names)
  new_table.to_csv(os.path.join(base_dir, 'trends_across_pace', 'poincare_sim_stats.csv'), index=False)
  test_table= pd.DataFrame(test_type_table, columns=test_type_names)
  test_table.to_csv(os.path.join(base_dir, 'trends_across_pace', 'poincare_sim_stats_test_types.csv'), index=False)
def re_formatting_peak(base_dir = os.path.join('Analysis')):
  df=pd.DataFrame()
  for speed in ['normal', 'slow', 'fast']:
    load_dir = os.path.join(base_dir, speed,'peaks_per_subject', 'gaussian_analysis')
    df_p = pd.read_csv(os.path.join(load_dir, 'combined_legs_test_t_and_wilcoxon.csv'))
    df_p['pace']=speed
    df = pd.concat([df, df_p])
  sensors = list(df['source'].unique())
  new_table = []
  test_type_data = []
  for sensor in sensors:
    row = [sensor]
    test_row = [sensor]
    column_names = ['sensor']
    subdf = df[(df['source']==sensor)]
    for speed in ['normal', 'slow', 'fast']:
      for metric in ['avg_peak',  'avg_range','avg_valley']:
        cond = (subdf['data']==metric)&(subdf['pace']==speed)
        column_names.append(speed+'_'+metric)
        assert len(subdf[cond]['p_value'].to_numpy())==1
        row.append(subdf[cond]['p_value'].iloc[0])
        test_row.append(subdf[cond]['test_type'].iloc[0])
    new_table.append(row)
    test_type_data.append(test_row)
  new_table= pd.DataFrame(new_table, columns=column_names)
  new_table.to_csv(os.path.join(base_dir,'trends_across_pace', 'peak_valley_sim_stats.csv'), index=False)
  test_type_table= pd.DataFrame(test_type_data, columns=column_names)
  test_type_table.to_csv(os.path.join(base_dir,'trends_across_pace', 'peak_valley_sim_stats_test_type.csv'), index=False)

  new_table = []
  test_type_data = []
  sensors = list(df[df['data']=='avg_gate_length']['source'].unique())
  for sensor in sensors:
    row = [sensor]
    test_row=[sensor]
    column_names = ['sensor']
    subdf = df[(df['source']==sensor)]
    for speed in ['normal', 'slow', 'fast']:
      for metric in ['avg_swing_index','avg_gate_length']:
        cond = (subdf['data']==metric)&(subdf['pace']==speed)
        column_names.append(speed+'_'+metric)
        assert len(subdf[cond]['p_value'].to_numpy())==1
        row.append(subdf[cond]['p_value'].iloc[0])
        test_row.append(subdf[cond]['test_type'].iloc[0])
    new_table.append(row)
    test_type_data.append(test_row)
  new_table= pd.DataFrame(new_table, columns=column_names)
  new_table.to_csv(os.path.join(base_dir,'trends_across_pace', 'gate_swing_sim_stats.csv'), index=False)
  test_type_table= pd.DataFrame(test_type_data, columns=column_names)
  test_type_table.to_csv(os.path.join(base_dir,'trends_across_pace', 'gate_swing_sim_stats_test_type.csv'), index=False)


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
    test_type_row = [sensor]
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
  test_type_data = []
  for sensor in sensors:
    subdf = df[(df['sensor']==sensor)]
    row = [sensor]
    test_row = [sensor]
    column_names = ['sensor' ]
    test_names = ['sensor' ]
    for speed in ['normal', 'slow', 'fast']:
      test_names.append(speed)
      test_row.append(subdf[subdf['pace']==speed]['test_type'].iloc[0])
      for head in data_columns:
        assert len(subdf[subdf['pace']==speed][head].to_numpy())==1,subdf[head].to_numpy()
        row.append(subdf[subdf['pace']==speed][head].iloc[0])
        column_names.append(speed+'_'+head)
    new_table.append(row)
    test_type_data.append(test_row)
  new_table= pd.DataFrame(new_table, columns=column_names)
  new_table.to_csv(os.path.join(base_dir,'trends_across_pace', 'paired_comparison_sim_stats.csv'), index=False)
  test_type_table= pd.DataFrame(test_type_data, columns=test_names)
  test_type_table.to_csv(os.path.join(base_dir,'trends_across_pace', 'paired_comparison_sim_stats_test_types.csv'), index=False)






