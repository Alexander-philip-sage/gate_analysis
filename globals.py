
all_motion_capture_columns = ["Rthigh_acc_x","Rthigh_acc_y","Rthigh_acc_z","Rthigh_vel_x","Rthigh_vel_y","Rthigh_vel_z","Lthigh_acc_x","Lthigh_acc_y","Lthigh_acc_z","Lthigh_vel_x","Lthigh_vel_y","Lthigh_vel_z","Rshank_acc_x","Rshank_acc_y","Rshank_acc_z","Rshank_vel_x","Rshank_vel_y","Rshank_vel_z","Lshank_acc_x","Lshank_acc_y","Lshank_acc_z","Lshank_vel_x","Lshank_vel_y","Lshank_vel_z","R_FP","L_FP"]
RIGHT_AVY_HEADER = 'Angular Velocity Y (rad/s).2'
LEFT_AVY_HEADER = 'Angular Velocity Y (rad/s).3'
COLUMNS_TO_GRAPH = ['Acceleration Y (m/s^2)',  'Angular Velocity Y (rad/s)',##right thigh
                    'Acceleration Y (m/s^2).1','Angular Velocity Y (rad/s).1',##left thigh
                    'Acceleration Y (m/s^2).2', RIGHT_AVY_HEADER,##right shank
                    'Acceleration Y (m/s^2).3',LEFT_AVY_HEADER ]##left shank

MOTION_CAPTURE_COLS = []
tmp = []
for val in COLUMNS_TO_GRAPH:
  tmp.append(val.replace('Y','X'))
  tmp.append(val.replace('Y','Z'))
COLUMNS_TO_GRAPH.extend(tmp)



COLUMNS_TO_AREA = {  'Acceleration Y (m/s^2)':'right thigh',  'Angular Velocity Y (rad/s)':'right thigh',##right thigh
                    'Acceleration Y (m/s^2).1':'left thigh','Angular Velocity Y (rad/s).1':'left thigh',##left thigh
                    'Acceleration Y (m/s^2).2':'right shank', RIGHT_AVY_HEADER:'right shank',##right shank
                    'Acceleration Y (m/s^2).3':'left shank',LEFT_AVY_HEADER:'left shank' }##left shank
tmp = list(COLUMNS_TO_AREA.items())
for val in tmp:
  COLUMNS_TO_AREA[val[0].replace('Y','X')] = val[1]
  COLUMNS_TO_AREA[val[0].replace('Y','Z')] = val[1]
COLUMNS_TO_LEG = {}
for k in list(COLUMNS_TO_AREA.keys()):
  COLUMNS_TO_LEG[k]=COLUMNS_TO_AREA[k].split(' ')[0]
COLUMNS_BY_SENSOR = [{'sensor':'thigh','right':'Acceleration Y (m/s^2)','left': 'Acceleration Y (m/s^2).1'},
                     {'sensor':'thigh','right':'Angular Velocity Y (rad/s)', 'left':'Angular Velocity Y (rad/s).1'},
                     {'sensor':'shank','right':'Acceleration Y (m/s^2).2','left':'Acceleration Y (m/s^2).3'},
                     {'sensor':'shank','right':RIGHT_AVY_HEADER, 'left':LEFT_AVY_HEADER},]
tmp = COLUMNS_BY_SENSOR.copy()
for d in tmp:
   COLUMNS_BY_SENSOR.append({'sensor':d['sensor'], 'right':d['right'].replace('Y','X'), 'left':d['left'].replace('Y','X')})
   COLUMNS_BY_SENSOR.append({'sensor':d['sensor'], 'right':d['right'].replace('Y','Z'), 'left':d['left'].replace('Y','Z')})
'''
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
from scipy.spatial import distance
#from scipy.signal import resample
from scipy.signal import correlate, find_peaks, butter, sosfilt
from scipy.interpolate import interp1d
from scipy.stats import shapiro, ttest_rel, wilcoxon
#! pip install statsmodels
import statsmodels.api as sm
from collections import defaultdict
import pickle
!pip install pyhrv
import pyhrv
import pyhrv.nonlinear as nl
import biosppy
from biosppy.signals.ecg import ecg
import matplotlib as mpl'''
DATA_DIR = 'raw_data'#'Text_File/Master_Data--TD_1-30_CSVFiles/Subjects_1-30-inw1_through_osw1'
GATE_CROSSING = -0.3
FREQUENCY = 128