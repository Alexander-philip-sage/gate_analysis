
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








