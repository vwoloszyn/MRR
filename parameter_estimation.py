import pandas as pd
import numpy as np
import NDCG as ndcg
import MHR as mhr
import sys
import types
ndcg.reload_package(mhr)
from sklearn.svm import SVR, LinearSVR
from sklearn.grid_search import GridSearchCV
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer
from time import time


file_out="data/parameters_estimation.csv"
global_parameters=[]

reviews_features = pd.read_csv('data/eletronic_sample_counts.csv.gz')



import scipy.optimize as optimization

quant_prod=380
sample=reviews_features.groupby('asin').size()[:quant_prod].to_dict().keys()

#df=reviews_features[reviews_features['asin'].isin(sample)]
#df_extra,ndcg_mhr = mhr.executeFromDf(df,0.9,-0.12)
#print ndcg_mhr
def func(params, xdata, ydata):
	
	global global_parameters
	
	record={}
	record['alpha']=params[0]
	record['beta']=params[1]

	print params[0]
	print params[1]
	df_extra,ndcg_mhr = mhr.executeFromDf(xdata,params[0],params[1])
	#return ndcg_mhr
	print "mean ndcg="+str(np.mean(ndcg_mhr))
	record['ndcg_mean']=np.mean(ndcg_mhr)
	global_parameters.append(record)

	pd.DataFrame(global_parameters).to_csv(file_out)
	return (ydata - ndcg_mhr)

#def func(x, a, b):
#    print a
#    print b
#    df_extra,ndcg_mhr = mhr.executeFromDf(x,a,b)
#    return ndcg_mhr

x0 = np.array([0.893, -0.1205])
ydata= np.ones(quant_prod,)
xdata = reviews_features[reviews_features['asin'].isin(sample)]
#print optimization.leastsq(func, x0, args=(xdata, ydata))
tplFinal1,success=optimization.leastsq(func,x0[:],args=(xdata,ydata),epsfcn=0.1)
#print optimization.curve_fit(func,  xdata, ydata, x0, bounds=((0,-1),(1,1)))
#print ndcg_mhr



print tplFinal1