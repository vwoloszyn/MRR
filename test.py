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


quant_prod=10

reviews_features = pd.read_csv('data/eletronic_sample_counts.csv.gz')
reviews_features['id_doc']=xrange(len(reviews_features))
sample=reviews_features.groupby('asin').size()[:quant_prod].to_dict().keys()
xdata = reviews_features[reviews_features['asin'].isin(sample)]




results,ndcg_mhr = mhr.executeFromDf(xdata)


## MRR scores for each document
# asin = if of product
# powerWithStar = score of revelnace of the document
# id_doc = id of document
print results[['asin','id_doc','powerWithStar']].values


