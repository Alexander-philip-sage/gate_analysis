import os
import glob
import datetime
import pandas as pd
import numpy as np

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

def z_score_directly(indoor_avg, indoor_std, outdoor_avg, outdoor_std):
  z_score  = (outdoor_avg-indoor_avg)/indoor_std
  cohens_d = (outdoor_avg-indoor_avg)/np.sqrt((indoor_std**2+outdoor_std**2)/2)
  return z_score, cohens_d

def extract_z_cohen_peak(sub_df_indoors, sub_df_outdoors, data_type):
  assert len(sub_df_indoors[sub_df_indoors['data']=='avg_'+data_type]['avg'])==1, sub_df_indoors[sub_df_indoors['data']=='avg_'+data_type]['avg']
  assert len(sub_df_outdoors[sub_df_outdoors['data']=='avg_'+data_type]['avg'])==1
  indoor_avg = sub_df_indoors[sub_df_indoors['data']=='avg_'+data_type]['avg'].iloc[0]
  outdoor_avg = sub_df_outdoors[sub_df_outdoors['data']=='avg_'+data_type]['avg'].iloc[0]
  indoor_std = sub_df_indoors[sub_df_indoors['data']=='avg_'+data_type]['std'].iloc[0]
  outdoor_std = sub_df_outdoors[sub_df_outdoors['data']=='avg_'+data_type]['std'].iloc[0]
  z_score, cohens_d = z_score_directly(indoor_avg, indoor_std, outdoor_avg, outdoor_std)
  return z_score, cohens_d


def peak_z_score_cohens_d(base_dir):
  df_peak_avgs=pd.DataFrame()
  for speed in ['normal', 'slow', 'fast']:
    load_dir = os.path.join(base_dir, speed,"peaks_per_subject", 'gaussian_analysis')
    df_p = pd.read_csv(os.path.join(load_dir,'combined_legs_test_shapiro_wilk.csv'))
    df_p['pace']=speed
    df_peak_avgs = pd.concat([df_peak_avgs, df_p])  
  list_sensors = list(df_peak_avgs['source'].unique())
  data = []
  print(df_peak_avgs['data'].unique())
  data_cohens = []
  for sensor in list_sensors:
    row_z = {"sensor": sensor}
    row_c = {"sensor": sensor}
    for speed in ['normal', 'slow', 'fast']:
      sub_df_indoors = df_peak_avgs[(df_peak_avgs['inout']=='indoors')&(df_peak_avgs['source']==sensor)&(df_peak_avgs['pace']==speed)]
      sub_df_outdoors = df_peak_avgs[(df_peak_avgs['inout']=='outdoors')&(df_peak_avgs['source']==sensor)&(df_peak_avgs['pace']==speed)]
      for data_type in ['peak','range', 'valley']:
        z_score, cohens_d = extract_z_cohen_peak(sub_df_indoors, sub_df_outdoors, data_type)
        row_z.update({speed+"_"+data_type+"_z_score":z_score})
        row_c.update({speed+"_"+data_type+"_cohens_d_score":cohens_d})
    data.append(row_z)
    data_cohens.append(row_c)
  df_z_score = pd.DataFrame(data)
  df_cohens_d = pd.DataFrame(data_cohens)
  df_z_score.to_csv(os.path.join(base_dir,"trends_across_pace","peak_z_score.csv"))
  df_z_score.to_csv(os.path.join(base_dir,"trends_across_pace","peak_cohens_d_score.csv"))

  data_c = []
  data_z = []  
  list_sensors = list(df_peak_avgs[df_peak_avgs['data']=='avg_gate_length']['source'].unique())
  for sensor in list_sensors:
    row_z = {"sensor": sensor}
    row_c = {"sensor": sensor}
    for speed in ['normal', 'slow', 'fast']:
      for data_type in ['swing_index','gate_length']:
        sub_df = df_peak_avgs[(df_peak_avgs['pace']==speed)&(df_peak_avgs['source']==sensor)]
        z_score, cohens_d = extract_z_cohen_peak(sub_df[sub_df['inout']=='indoors'], sub_df[sub_df['inout']=='outdoors'], data_type)
        row_z.update({speed+"_"+data_type+"_z_score":z_score})
        row_c.update({speed+"_"+data_type+"_cohens_d_score":cohens_d})
    data_c.append(row_c)
    data_z.append(row_z)
  table_c= pd.DataFrame(data_c)
  table_z= pd.DataFrame(data_z)
  table_c.to_csv(os.path.join(base_dir,'trends_across_pace', 'gate_swing_cohens_d.csv'), index=False)
  table_z.to_csv(os.path.join(base_dir,'trends_across_pace', 'gate_swing_z_score.csv'), index=False)

