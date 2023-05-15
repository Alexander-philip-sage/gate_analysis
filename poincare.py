import pyhrv
import pyhrv.nonlinear as nl
import biosppy
#from biosppy.signals.ecg import ecg
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

print("pyhrv package version",pyhrv.__version__)
print("biosppy package version",biosppy.__version__)

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