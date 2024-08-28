from scipy.signal import correlate, find_peaks, butter, sosfilt
from typing import List, Tuple
import numpy as np
def avg_std_gate_lengths(zero_crossings: List[Tuple[int]]):
  points_p_gate = [x[1]-x[0] for x in zero_crossings]
  points_p_gate = np.array(points_p_gate)
  return  points_p_gate.mean() , points_p_gate.std(), points_p_gate.max(), points_p_gate.min()

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