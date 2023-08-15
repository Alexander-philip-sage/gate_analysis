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
    print(sensor)
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








