from datetime											import date, timedelta
from matplotlib											import pyplot as plt
from numpy 												import array
from scipy												import stats
from os.path 										 	import exists as os_path_exists, \
															join as os_path_join
from os 												import makedirs as os_mkdir
from statsmodels.distributions.empirical_distribution 	import ECDF

#import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas 			 as pd

#plots
def cdf_plt(x, legend, xlbl, ttl, fl_nm):
	ecdf = ECDF(x)
	ecdf_x, ecdf_y = ecdf.x, ecdf.y
	make_plot(ecdf_x, ecdf_y, legend, xlbl, "CDF", ttl, fl_nm)


def surv_plt(x, legend, xlbl, ttl, fl_nm):
	ecdf = ECDF(x)
	ecdf_x, ecdf_y = ecdf.x, 1.0 - ecdf.y
	make_plot(ecdf_x, ecdf_y, legend, xlbl, "Survival plot", 
					ttl, fl_nm, is_mark=True)
	

def make_bar(xticks, y, ylbl, ttl, fl_nm):
	fig = plt.figure()
	y_pos = np.arange(len(xticks)) 
	plt.bar(y_pos, y, align='center', alpha=0.5)
	plt.xticks(y_pos, xticks, rotation='vertical')
	plt.ylabel(ylbl)
	plt.title(ttl)
	fig.set_size_inches(15,9)
	plt.savefig(fl_nm+".png")


def writable(dir_nm, fl_nm):
	if not os_path_exists(dir_nm):
		os_mkdir(dir_nm)
	return os_path_join(dir_nm, fl_nm)


def subplots(fig, xlbl, ylbl, ttl, fl_nm):
	plt.xlabel(xlbl)
	plt.ylabel(ylbl)
	plt.title(ttl)
	fig.set_size_inches(15,9)
	plt.savefig(fl_nm+".png")
	plt.close()


def make_hist(x, xlbl, ylbl, ttl, fl_nm):
	fig, ax = plt.figure(), plt.subplot()
	plt.hist(x)
	subplots(fig, xlbl, ylbl, ttl, fl_nm)


def make_2hist(x1, x2, xlbl, ylbl, x1leg, x2leg, ttl, fl_nm):
	fig, ax = plt.figure(), plt.subplot()
	plt.hist(x1, alpha=0.5, label=x1leg, normed=True)
	plt.hist(x2, alpha=0.5, label=x2leg, normed=True)
	plt.legend()
	subplots(fig, xlbl, ylbl, ttl, fl_nm)


def make_scatter(x, y, xlbl, ylbl, ttl, fl_nm):
	# x, y = array(x), array(y)
	# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
	# line = slope*x+intercept
	fig, ax = plt.figure(), plt.subplot()
	plt.scatter(x, y)
	subplots(fig, xlbl, ylbl, ttl, fl_nm)


def make_plot(x, y, legend, xlbl, ylbl, ttl, fl_nm, is_mark=False):
	fig, ax = plt.figure(), plt.subplot()
	#To plot multiple plots
	# if type(x) == list():
	# 	for x_, y_, leg_ in zip(x,y,legend):
	# 		plt.plot(x_, y_, label=leg_)
	if is_mark:
		plt.plot(x, y, marker='o')
	else:
		plt.plot(x, y, label=legend)
	subplots(fig, xlbl, ylbl, ttl, fl_nm)


def make_2plot(x1, x2, y1, y2, x1leg, x2leg, xlbl, ylbl, ttl, fl_nm):
	fig, ax = plt.figure(), plt.subplot()
	plt.plot(x1, y1, label=x1leg)
	plt.plot(x2, y2, label=x2leg)
	plt.legend()
	subplots(fig, xlbl, ylbl, ttl, fl_nm)


def plt_mat(mat, xlbl, ylbl, ttl, fl_nm, m_aspect=0.1, m_cmap="binary"):
	fig, ax = plt.subplots(figsize=(20,20))
	ax.matshow(mat, aspect=m_aspect, cmap=m_cmap)
	xticks = list(mat.columns.values)
	ax.set_xticklabels(['']+xticks)
	plt.xlabel(xlbl, fontsize=20)
	plt.ylabel(ylbl, fontsize=20)
	plt.title(ttl, fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=20)
	plt.tick_params(axis='both', which='minor', labelsize=20)
	plt.savefig(fl_nm)
	plt.close()
	# subplots(fig, xlbl, ylbl, ttl, fl_nm)


def rangefortime(frst_dt, last_dt, timedelta):
	'''Since we do not have data for July 
	this iteration of dates skips July'''
	curr_dt = frst_dt
	while curr_dt < last_dt:
		if (curr_dt+timedelta).month==7:
			yield curr_dt, date(day=30, month=06, year=2017)
			curr_dt = date(day=01, month=8, year=2017)
		if (curr_dt.month == 7):
			curr_dt = date(day=01, month=8, year=2017)
		yield curr_dt, curr_dt+timedelta
		curr_dt += timedelta


def mk_latex_frm_dict(my_dict, col_nms, fl_out,	
								istranspose=False):
	'''Converts the dictinary into a 
	latex table'''
	if istranspose: tbl = pd.DataFrame.from_dict(my_dict).T
	else: tbl = pd.DataFrame.from_dict(my_dict)
	tbl.rename(columns=col_nms, inplace=True)
	tbl.round(3).to_latex(fl_out)
