##https://sciendo.com/article/10.2478/slgr-2013-0031
#!pip install pyhrv
import pyhrv
import pyhrv.nonlinear as nl
import biosppy
from biosppy.signals.ecg import ecg
import matplotlib as mpl
import os, datetime, pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from globals import RIGHT_AVY_HEADER, LEFT_AVY_HEADER
from globals import COLUMNS_TO_GRAPH, COLUMNS_TO_AREA, COLUMNS_TO_LEG, COLUMNS_BY_SENSOR
from similarity import combine_legs_single_subject
import pandas as pd
import glob
from scipy.stats import shapiro, ttest_rel, wilcoxon
def poincare(nni=None,
			 rpeaks=None,
			 show=True,
			 figsize=None,
			 ellipse=True,
			 vectors=True,
			 legend=True,
			 marker='o',
       title = 'Poincare Plot',
			 mode='normal',
			 xlim=None,
			 ylim=None):
	"""Creates Poincaré plot from a series of NN intervals or R-peak locations and derives the Poincaré related
	parameters SD1, SD2, SD1/SD2 ratio, and area of the Poincaré ellipse.
	References: [Tayel2015][Brennan2001]
	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s]
	rpeaks : array
		R-peak times in [ms] or [s]
	show : bool, optional
		If true, shows Poincaré plot (default: True)
	show : bool, optional
		If true, shows generated plot
	figsize : array, optional
		Matplotlib figure size (width, height) (default: (6, 6))
	ellipse : bool, optional
		If true, shows fitted ellipse in plot (default: True)
	vectors : bool, optional
		If true, shows SD1 and SD2 vectors in plot (default: True)
	legend : bool, optional
		If True, adds legend to the Poincaré plot (default: True)
	marker : character, optional
		NNI marker in plot (default: 'o')
		mode : string, optional
	Return mode of the function; available modes:
		'normal'	Returns frequency domain parameters and PSD plot figure in a ReturnTuple object
		'dev'		Returns frequency domain parameters, frequency and power arrays, no plot figure
	Returns (biosppy.utils.ReturnTuple Object)
	------------------------------------------
	[key : format]
		Description.
	poincare_plot : matplotlib figure object
		Poincaré plot figure
	sd1 : float
		Standard deviation (SD1) of the major axis
	sd2 : float, key: 'sd2'
		Standard deviation (SD2) of the minor axis
	sd_ratio: float
		Ratio between SD2 and SD1 (SD2/SD1)
	ellipse_area : float
		Area of the fitted ellipse
	"""
	# Check input values
	nn = pyhrv.utils.check_input(nni, rpeaks)/1000

	# Prepare Poincaré data
	x1 = np.asarray(nn[:-1])
	x2 = np.asarray(nn[1:])

	# SD1 & SD2 Computation
	sd1 = np.std(np.subtract(x1, x2) / np.sqrt(2))
	sd2 = np.std(np.add(x1, x2) / np.sqrt(2))

	# Area of ellipse
	area = np.pi * sd1 * sd2

	# Dev:
	# Output computed SD1 & SD2 without plot
	if mode == 'dev':
		# Output
		args = (sd1, sd2, sd2 / sd1, area)
		names = ('sd1', 'sd2', 'sd_ratio', 'ellipse_area')
		return biosppy.utils.ReturnTuple(args, names)

	# Normal:
	# Same as dev but with plot
	if mode == 'normal':
		if figsize is None:
			figsize = (6, 6)
		fig = plt.figure(figsize=figsize)
		fig.tight_layout()
		ax = fig.add_subplot(111)

		ax.set_title(title)
		ax.set_ylabel('x(n+1)')
		ax.set_xlabel('x(n)')
		if xlim:
			ax.set_xlim(xlim)
		else:
			ax.set_xlim([np.min(nn) - 0.5, np.max(nn) + 0.5])
		if ylim:
			ax.set_ylim(ylim)
		else:
			ax.set_ylim([np.min(nn) - 0.5, np.max(nn) + 0.5])
		ax.grid()
		ax.plot(x1, x2, 'r%s' % marker, markersize=2, alpha=0.5, zorder=3)

		# Compute mean NNI (center of the Poincaré plot)
		nn_mean = np.mean(nn)

		# Draw poincaré ellipse
		if ellipse:
			ellipse_ = mpl.patches.Ellipse((nn_mean, nn_mean), sd1 * 2, sd2 * 2, angle=-45, fc='k', zorder=1)
			ax.add_artist(ellipse_)
			ellipse_ = mpl.patches.Ellipse((nn_mean, nn_mean), sd1 * 2 - 1, sd2 * 2 - 1, angle=-45, fc='lightyellow', zorder=1)
			ax.add_artist(ellipse_)

		# Add poincaré vectors (SD1 & SD2)
		if vectors:
			arrow_head_size = 0.1
			na = 2
			a1 = ax.arrow(
				nn_mean, nn_mean, (-sd1 + na) * np.cos(np.deg2rad(45)), (sd1 - na) * np.sin(np.deg2rad(45)),
				head_width=arrow_head_size, head_length=arrow_head_size, fc='g', ec='g', zorder=4, linewidth=1.5)
			a2 = ax.arrow(
				nn_mean, nn_mean, (sd2 - na) * np.cos(np.deg2rad(45)), (sd2 - na) * np.sin(np.deg2rad(45)),
				head_width=arrow_head_size, head_length=arrow_head_size, fc='b', ec='b', zorder=4, linewidth=1.5)
			a3 = mpl.patches.Patch(facecolor='white', alpha=0.0)
			a4 = mpl.patches.Patch(facecolor='white', alpha=0.0)
			ax.add_line(mpl.lines.Line2D(
				(min(nn), max(nn)),
				(min(nn), max(nn)),
				c='b', ls=':', alpha=0.6))
			ax.add_line(mpl.lines.Line2D(
				(nn_mean - sd1 * np.cos(np.deg2rad(45)) * na, nn_mean + sd1 * np.cos(np.deg2rad(45)) * na),
				(nn_mean + sd1 * np.sin(np.deg2rad(45)) * na, nn_mean - sd1 * np.sin(np.deg2rad(45)) * na),
				c='g', ls=':', alpha=0.6))

			# Add legend
			if legend:
				ax.legend(
					[a1, a2, a3, a4],
					['SD1: %.3f' % sd1, 'SD2: %.3f' % sd2, 'Area: %.3f' % area, 'SD1/SD2: %.3f' % (sd1/sd2)],
					framealpha=1)

		# Show plot
		if show:
			plt.show()

		# Output
		args = (fig, sd1, sd2, sd2/sd1, area)
		names = ('poincare_plot', 'sd1', 'sd2', 'sd_ratio', 'ellipse_area')
		return biosppy.utils.ReturnTuple(args, names)
	
def graph_poincare_per_leg(data_lookup, metadata, zero_crossing_lookup, sensor, save_dir):
  ''' for single sensor, for indoors and out, for each subject do poincare on the
  avg signal '''
  area=COLUMNS_TO_AREA[sensor]
  save_dir = os.path.join(save_dir,sensor.replace(os.path.sep, "-").replace(" ", '')+'_'+area.replace(' ',"_"))
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  data_poincare=[]
  print("pace",metadata['pace'].unique())
  for inout in ['indoors', 'outdoors']:
    for subjectID in  metadata['subjectID'].unique():
      all_gates = aggregate_single_subject(data_lookup, metadata, zero_crossing_lookup, sensor, inout , subjectID)
      avg =all_gates.mean(axis=0)
      title = "{} {} {} subjectID: {}".format(sensor,area, inout,subjectID)
      figname = "{}_{}_{}_{}.png".format(sensor.replace(os.path.sep, "-"),area, inout,subjectID)
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
  save_dir = os.path.join(save_dir_m,sensor.replace(os.path.sep, "-").replace(" ", '').replace("^",'-'))
  #print("saving at",save_dir)
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  data_poincare=[]
  for inout in ['indoors', 'outdoors']:
    avg_signal = combined_legs[sensor][inout]['avg']
    title = "{} {}".format(sensor, inout)
    figname = "{}_{}.png".format(sensor.replace(os.path.sep, "-"), inout)
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
def graph_poincare_comb_leg(combined_legs, save_dir_m):
  print("pace",metadata['pace'].unique()[0])
  for sensor in list(combined_legs.keys()):
    graph_poincare_comb_leg_per_sensor(combined_legs, sensor, save_dir_m)

def graph_poincare_comb_leg_per_sensor_per_subject(metadata,data_lookup,zero_crossing_lookup, sensor_cols, save_dir_m ,ylim=None, xlim=None):
  ''' for single sensor - combined legs, for indoors and out, for each subject do poincare on the
  avg signal '''
  sensor_name = sensor_cols['left'].replace(".1",' '+sensor_cols['sensor']).replace(".3",' '+sensor_cols['sensor'])
  save_dir = os.path.join(save_dir_m,sensor_name.replace(os.path.sep,'-'))
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  rows = []
  for subjectID in  metadata['subjectID'].unique():
      indoors, outdoors = combine_legs_single_subject(sensor_name,data_lookup, metadata, zero_crossing_lookup, sensor_cols, subjectID )
      for signal, inout in [(indoors,'indoors'), (outdoors, 'outdoors')]:
        title = "{} {} subject:{}".format(sensor_name, inout, subjectID)
        figname = "{}_{}_subject{}.png".format(sensor_name.replace(os.path.sep,'-'), inout, subjectID)

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
def calculate_poincare_stats(alpha = 0.05):
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


