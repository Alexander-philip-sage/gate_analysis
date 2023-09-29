
from globals import COLUMNS_TO_AREA, FREQUENCY
import numpy as np
import random
from load_data import extract_trial_data
from gate_crossings import label_axis, continuous_gate_crossings
from peaks_valley_swing import find_lowest_valley, max_peak

def raw_summary_graph_multi_gate(subjectID, sensor, gate_ind, zero_crossings, df_raw, summary_dir):
  ##graphing first gate crossing
  start_gate, end_gate = zero_crossings[gate_ind][0], zero_crossings[gate_ind+5][1]
  print('indices', start_gate, end_gate)
  points_raw = df_raw[sensor].values[start_gate:end_gate]
  t = np.array([x for x in range(len(points_raw))])/FREQUENCY
  fig, ax1 = plt.subplots(1, 1, figsize=(12,8))
  ax1.plot(t, points_raw, color='black')
  ax1.set_title('Raw signal')
  ax1.set_xlabel("time (s)")
  ax1.set_ylabel("rad/s")
  _=fig.suptitle("Subject " +str(subjectID)+" " + sensor+ " " + COLUMNS_TO_AREA[sensor])
  ax1.grid(visible=True)
  sensor = sensor.replace("/", "-")
  fig.savefig(os.path.join(summary_dir, f'subject_{subjectID}_0rawmultigate_{sensor}.png'))

def raw_summary_graph(subjectID, sensor, zero_crossings, df_raw, summary_dir):
  ##graphing first gate crossing
  start_gate, end_gate = zero_crossings[0]
  print('indices', start_gate, end_gate)
  points_raw = df_raw[sensor].values[start_gate:end_gate]
  t = np.array([x for x in range(len(points_raw))])/FREQUENCY
  fig, ax1 = plt.subplots(1, 1, figsize=(12,8))
  ax1.plot(t, points_raw, color='black')
  ax1.set_title('Raw signal')
  ax1.set_xlabel("time (s)")
  ax1.set_ylabel("rad/s")
  _=fig.suptitle("Subject " +str(subjectID)+" " + sensor+ " " + COLUMNS_TO_AREA[sensor])
  ax1.grid(visible=True)
  sensor = sensor.replace("/", "-")
  fig.savefig(os.path.join(summary_dir, f'subject_{subjectID}_0raw_{sensor}.png'))


def graph_summary_butterworth(subjectID,file, sensor, gate_ind, df_filtered, side, summary_dir, zero_crossing_lookup, N=4, Wn=20):
  zero_crossings = zero_crossing_lookup[file][side]
  start_gate, end_gate = zero_crossings[gate_ind]
  print('indices', start_gate, end_gate)
  points_filtered = df_filtered[sensor].values[start_gate:end_gate]
  t = np.array([x for x in range(len(points_filtered))])/FREQUENCY

  fig, ax2 = plt.subplots(1, 1, figsize=(12,8))
  ax2.plot(t, points_filtered, color='black')
  ax2.set_xlabel("time (s)")
  ax2.set_ylabel("rad/s")
  ax2.set_title(" Butterworth Filter " +" N = "+str(N)+", Wn = "+str(Wn)+"Hz")
  _=fig.suptitle("Subject " +str(subjectID)+" " + sensor+ " " + COLUMNS_TO_AREA[sensor])
  ax2.grid(visible=True)
  sensor = sensor.replace("/", "-")
  fig.savefig(os.path.join(summary_dir, f'subject_{subjectID}_1butterworth_{sensor}.png'))

def graph_gate_crossing_summary(subjectID, sensor, df_filtered, gate_ind, zero_crossings, summary_dir):
  start_gate, end_gate = zero_crossings[gate_ind][0], zero_crossings[gate_ind+5][1]
  filtered_values = df_filtered[sensor].values
  print('indices', start_gate, end_gate)
  points_gates=filtered_values[start_gate:end_gate]
  t = np.array([x for x in range(len(points_gates))])/FREQUENCY

  ct_gate_crossing_marks = 5
  y_index_points = [zero_crossings[gate_ind+i][1] for i in range(ct_gate_crossing_marks)]
  x_index_points = [zero_crossings[gate_ind+i][1]-zero_crossings[gate_ind][0] for i in range(ct_gate_crossing_marks)]
  x_index_points[-1] = x_index_points[-1] -1
  x_points = [t[i] for i in x_index_points]
  y_points = [filtered_values[i] for i in y_index_points]

  fig, ax3 = plt.subplots(1, 1, figsize=(12,8))
  ax3.plot(t, points_gates, color='black')
  ax3.scatter(x_points, y_points, color='r', s=200)
  ax3.set_xlabel("time (s)")
  ax3.set_ylabel("rad/s")
  ax3.set_title(" Gate Detection ")
  _=fig.suptitle("Subject " +str(subjectID)+" " + sensor+ " " + COLUMNS_TO_AREA[sensor])
  ax3.grid(visible=True)
  sensor = sensor.replace("/", "-")
  fig.savefig(os.path.join(summary_dir, f'subject_{subjectID}_2gate_detection_{sensor}.png'))

def graph_summary_avg(subjectID, sensor, indoors, data_lookup, metadata, zero_crossing_lookup, summary_dir):
  all_gates = aggregate_single_subject(data_lookup, metadata, zero_crossing_lookup, sensor, indoors , subjectID)
  avg = all_gates.mean(axis=0)
  std = all_gates.std(axis=0)
  fig, ax = plt.subplots(figsize=(12,8))
  ax.plot(avg, color='black')
  label_axis(sensor, ax)
  ax.grid(visible=True)
  ax.fill_between(np.arange(0,100), avg-std, avg+std, alpha=0.4, color='gray')
  ax.set_title(" Averaged Signal ")
  _=fig.suptitle("Subject " +str(subjectID)+" " + sensor+ " " + COLUMNS_TO_AREA[sensor])
  sensor = sensor.replace("/", "-")
  fig.savefig(os.path.join(summary_dir, f'subject_{subjectID}_3averaged_signal_{sensor}.png'))
  return avg, std

def graph_summary_avg_marked(subjectID, sensor, avg, std, summary_dir):
  fig, ax = plt.subplots(figsize=(12,8))
  ax.plot(avg, color='black')
  tup1, tup2 = max_peak(avg, edge_start=3), find_lowest_valley(avg)
  ax.scatter([tup1[0], tup2[0]],[tup1[1], tup2[1]], color='r', s=200)
  label_axis(sensor, ax)
  ax.grid(visible=True)
  ax.fill_between(np.arange(0,100), avg-std, avg+std, alpha=0.4, color='gray')
  ax.set_title(" Averaged Signal ")
  _=fig.suptitle("Subject " +str(subjectID)+" " + sensor+ " " + COLUMNS_TO_AREA[sensor])
  sensor = sensor.replace("/", "-")
  fig.savefig(os.path.join(summary_dir, f'subject_{subjectID}_4labeled_peak_valley_{sensor}.png'))


def make_summary_plots(metadata,zero_crossing_lookup,data_lookup, subjectID =None, sensor=None):
  if not subjectID:
    subjectID = metadata['subjectID'].values[random.randint(0,metadata.shape[0]-1)]
  #'20221030-100150-subject_21inw1.csv'
  file =metadata['filename'][(metadata['subjectID']==subjectID) & (metadata['inout']=='indoors') & (metadata['trial']==1)].iloc[0]
  subjectID, indoors, speed , _, _  = extract_trial_data(file)#
  print("file", file)
  print("subject", subjectID)
  print("inout", indoors, "speed", speed)
  if not sensor:
    sensor = LEFT_AVY_HEADER

  print("sensor", sensor)
  zero_crossings_left,zero_crossings_right, df_raw = continuous_gate_crossings(file)
  side = COLUMNS_TO_LEG[sensor]
  if side=='right':
    zero_crossings=zero_crossings_right
  elif side=='left':
    zero_crossings=zero_crossings_left
  gate_ind = np.random.randint(1,(len(zero_crossing_lookup[file][side])-1)) # random.randint(0,len(zero_crossings)-2)

  ##filtered data
  ##zero crossings loaded from pickle
  ##graphing first gate crossing
  raw_summary_graph_multi_gate(subjectID, sensor,gate_ind, zero_crossings, df_raw)
  #raw_summary_graph(subjectID, sensor, zero_crossings, df_raw)
  #graph_summary_butterworth(subjectID,file, sensor, gate_ind, data_lookup[file], side)
  graph_gate_crossing_summary(subjectID, sensor, data_lookup[file], gate_ind,zero_crossing_lookup[file][side] )
  avg, std = graph_summary_avg(subjectID, sensor, indoors)
  graph_summary_avg_marked(subjectID, sensor, avg, std)
