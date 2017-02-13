#!/mu/sdk/anaconda/bin/python

import sys
import pandas as pd
import numpy as np
from matplotlib.dates import datestr2num,num2date
import sklearn.neighbors as nb

flim = 1.65

full = pd.read_table(sys.stdin,sep='\t',header=None)
full.columns = ['tool_name','traveler_step','part_type',\
	'recipe','start_time','step_id','aggregation_name',\
	'step_occurence','sensor','value','state','event_code']
compiled = list()
for tool in full.tool_name.unique():
	raw = full[full.tool_name == tool].sort_values(['start_time'])
	if raw[raw.state != r'\N'].shape[0] == 0:
		last_pm = raw.start_time.values[0]
	else:
		last_pm = raw[raw.state != r'\N'].start_time.values[-1]
	df_v = raw[raw.value != r'\N']
	df_v['value'] = df_v['value'].astype('float64')
	upper = np.mean(df_v.value) + 3*np.std(df_v.value)
	lower = np.mean(df_v.value) - 3*np.std(df_v.value)
	df_v = df_v[(df_v.value < upper) & (df_v.value > lower)]

	part_A = df_v[df_v.start_time < last_pm].groupby(['tool_name','start_time'])\
		.value.mean().reset_index()
	part_A = part_A.sort_values('start_time')
	part_B = df_v[df_v.start_time > last_pm].groupby(['tool_name','start_time'])\
		.value.mean().reset_index()
	part_B = part_B.sort_values('start_time')

	xd_B = np.array([datestr2num(d) for d in part_B.start_time])
	if xd_B.shape[0] < 1:
		continue
	smth = nb.KNeighborsRegressor(min(20,xd_B.shape[0])).fit(xd_B.reshape(-1,1),part_B.value)
	b_dat = xd_B[np.argmax(smth.predict(xd_B.reshape(-1,1)))]  # locate peak        
	if b_dat == xd_B[-1]:
		p_dat = b_dat.ravel()
	else:
		p_dat = np.arange(b_dat,xd_B[-1]+1)
	if p_dat.shape[0] < 3:
		smoothed = smth.predict(p_dat.reshape(-1,1))
		mu = 0
		sig = 0
	else:
		smoothed = smth.predict(p_dat.reshape(-1,1))
		incre = np.diff(smoothed)[-100:]
		mu = np.mean(incre)
		sig = np.std(incre)

	def iGpdf(x,mu=-mu,sig=sig,L=smoothed[-1]-flim):
		return L/np.sqrt(2*np.pi*sig**2*x**3)*np.exp(-(L-x*mu)**2/(2*sig**2*x))

	i_dat = np.arange(1,180)
	p_den = iGpdf(i_dat)
	if (p_den.min() < 0) or (np.isnan(p_den.min())): p_den = np.zeros(i_dat.shape[0])
	f_dat = np.floor(p_dat[-1]) + i_dat

	fs_dat = [num2date(d).strftime('%Y-%m-%d %H:%M:%S') for d in f_dat]
	part_C = pd.DataFrame(np.c_[fs_dat,p_den],columns=['start_time','EOL_density'])
	part_C['tool_name'] = tool
	part_B['smoothed'] = pd.Series(smth.predict(xd_B.reshape(-1,1)))

	res = pd.concat([part_A,part_B,part_C],ignore_index=True)
	res['limit'] = pd.Series(np.ones(res.shape[0])*flim)
	eol_index = np.argmax(p_den)
	if eol_index > 0:
		res['EOL_date'] = fs_dat[eol_index]
	else:
		res['EOL_date'] = np.nan
	compiled.append(res)

Res = pd.concat(compiled,ignore_index=True)
Res.to_csv(sys.stdout,index=False,header=False)

