from load_data import load_data
from globals import LEFT_AVY_HEADER, RIGHT_AVY_HEADER, COLUMNS_TO_LEG
#########SUMMARIZING DATA REMOVED/MISSING###############
def find_missing(condition, name, metadata, PACE):
  one_to_thirty = set([x for x in range(1,31)])
  subjects_found = set(metadata[condition]['subjectID'].unique())
  subjects_missing = one_to_thirty.difference(subjects_found)
  print(PACE)
  print(name)
  print("subjects missing", subjects_missing)

def count_data_points_saved(metadata, zero_crossing_lookup):
  columns = ['filename', 'subjectID', 'inout', 'pace', 'trial', 'sensor', 'raw_data_points', 'good_data_points', 'percent_good']
  data = []
  for i, row in metadata.iterrows():
    filename = row.filename
    df_raw = load_data(filename)
    raw_data_points = df_raw.shape[0]
    for sensor in [LEFT_AVY_HEADER, RIGHT_AVY_HEADER]:
      new_row = {}
      #print("sensor", sensor)
      #t=data_lookup[filename][sensor]
      #t=t.dropna()
      #raw_data_points = len(t)
      zero_crossings=zero_crossing_lookup[filename][COLUMNS_TO_LEG[sensor]]
      gates = [tup[1]-tup[0] for tup in zero_crossings]
      data_points_saved = sum(gates)
      if raw_data_points<data_points_saved:
        print(zero_crossings)
        print(data_points_saved)
        print(raw_data_points)
        print("max", max(gates))
        return
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
