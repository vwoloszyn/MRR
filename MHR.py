# -*- coding: utf-8 -*-

import sys, gzip, json, datetime
import nltk
import pandas as pd
import random
import os, sys, gzip, json, datetime
import pandas as pd
import numpy as np
import re
import string 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import math
import matplotlib.pyplot as plt




matrix = []
bin_matrix = []
scores = []


def helpf(x): 
	try:
		pos = x['helpful'].replace("[","").replace("]","").split(',')[0]
		neg = x['helpful'].replace("[","").replace("]","").split(',')[1]
		tot = x['helpful'].replace("[","").replace("]","").split(',')[1]
		return float ( float(pos) /  float(tot) )
	except:
		return 0

def tot(x): 
	try:
		return x['helpful'].replace("[","").replace("]","").split(',')[1]
	except:
		return 0
    

def remove_stop(value):
	for w in stopwords.words('english'):
		value= value.replace(w,"")
		#value = re.sub(w, "", value)
	return value

def compute_distance_concept(comments):
	comment_count = len(comments)
	matrix = np.zeros((comment_count,comment_count ))

	for row in range(comment_count):
		for col in range(comment_count):
			matrix[row, col] = compute_distance_wives(comments[row].split(),comments[col].split())
			#print comments[row]

	return matrix


def compute_distance_wives(sentence1, sentence2):


    # RETURN METODO_WIVES_AQUI()   
    EPSILON = 0.0000000000000001
    result = 0
    
    # identify common words
    common_words = frozenset(sentence1) & frozenset(sentence2)
    
    if len(sentence1) > len(sentence2): 
        maxLen = len(sentence1); 
        minLen = len(sentence2) 
    else: 
        maxLen = len(sentence2); 
        minLen = len(sentence1) 
    
    # calculates similarity
    wordWeightMax = 0; wordWeightMin = 0;
    for term in common_words:
        if wordWeightMax < len(term): wordWeightMax = len(term)
        if wordWeightMin > len(term): wordWeightMin = len(term)
        negationWordWeightMax = 1 - wordWeightMax;
        negationWordWeightMin = 1 - wordWeightMin;
            
        c1 = 1 if wordWeightMin == 0 else wordWeightMax / wordWeightMin;
        c2 = 1 if wordWeightMax == 0 else wordWeightMin / wordWeightMax;
        c3 = 1 if negationWordWeightMin == 0 else negationWordWeightMax / negationWordWeightMin;
        c4 = 1 if negationWordWeightMax == 0 else negationWordWeightMin / negationWordWeightMax;

        m1 = min(min(c1, c2), 1);
        m2 = min(min(c3, c4), 1);

        result += 0.5*(m1+m2);
    
    result =math.fabs(result / (minLen + maxLen - len(common_words) + EPSILON));
#     print(result)
    return result;


def clear_string(value):
	
	for c in string.punctuation:
		#print c
		value= value.replace(c,"")

	return value.lower()




class PageRank_NoTeleport:
    "Power iteration but does not address Spider trap problem or Dead end problem "
    epsilon = 0.0001

    def __init__(self, beta=0.85, epsilon=0.00001):
        self.epsilon = epsilon

    def distance(self, v1, v2):
        v = v1 - v2
        v = v * v
        return np.sum(v)

    def compute(self, G):
        "G is N*N matrix where if j links to i then G[i][j]==1, else G[i][j]==0"
        N = len(G)
        d = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if (G[j, i] == 1):
                    d[i] += 1

        r0 = np.zeros(N, dtype=np.float32) + 1.0 / N
        # construct stochastic M
        M = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if G[j, i] == 1:
                    M[j, i] = 1.0 / d[i]
        while True:
            r1 = np.dot(M, r0)
            dist = self.distance(r1, r0)
            if dist < self.epsilon:
                break
            else:
                r0 = r1

        return r1

def create_matrix_sentence(comments):

	comment_count = len(comments)
	tfidf = TfidfVectorizer().fit_transform(comments)
	matrix_sentences = (tfidf * tfidf.T).A
	return 1 - matrix_sentences
    


def create_matrix_stars(stars):
	matrix_stars=np.zeros((len(stars), len(stars)))

	for row in range(len(stars)):
		for col in range(len(stars)):
			matrix_stars[row][col]= abs(stars[row]-stars[col])
			#print "distance" + str(matrix_stars[row][col])


	matrix_stars = matrix_stars/5
	matrix_stars=matrix_stars
	return matrix_stars



def getMostSalientWithStar(comments, stars, comments_count,alpha=0.9,beta=-0.12):

	global matrix
	global bin_matrix
	global scores

	bin_matrix=[]
	matrix =[]
	scores = []
	#threshold = beta

	#alpha=0.9


	matrix_sentences = create_matrix_sentence(comments )
	#matrix_sentences = compute_distance_concept(comments)

	matrix_stars = create_matrix_stars(stars)

	



	matrix = np.array(alpha*matrix_sentences) + ((1-alpha)* np.array(matrix_stars))
	matrix = matrix/np.amax(matrix)

	#matrix_sentences =alpha* matrix_sentences
	#matrix_stars=(1-alpha)* matrix_stars
	#matrix = 2* ((matrix_sentences*matrix_stars)/(matrix_sentences+matrix_stars))

	bin_matrix=np.zeros((len(comments), len(comments)))
	if (beta>1):
		threshold = beta -1
	else:
		threshold = np.mean(matrix)*(1+beta)
	#print threshold
	for row in range(len(matrix)):
		for col in range(len(matrix)):
			if matrix[row, col] < threshold:
			    bin_matrix[row, col] = 1.0
			    #degrees[row] += 1
			else:
			    bin_matrix[row, col] = 0

	
	#print matrix
	pr1 = PageRank_NoTeleport()
	scores = pr1.compute(bin_matrix)
	
	#print scores
	
	return np.array(scores)


def cal_metrics(a,b,k):
	precision=0
	recall=0
	f1=0
	#print b
	try:
		count=0
		ind=1
		for i in (-np.array(b)).argsort()[:k]:
			if (i in (-np.array(a)).argsort()[:k]):
				#count=count+1
				precision=precision+1
			ind=ind+1
		
		precision=float(precision)/float(k)
	except:
		precision=0
	#precision=len(np.intersect1d(-(np.array(a)).argsort()[:n_ref], -(np.array(b)).argsort()[:n_feat] ))/float(n_feat)

	#recall=len(np.intersect1d(-(np.array(a)).argsort()[:n_ref], -(np.array(b)).argsort()[:n_feat] ))/float(n_ref)

	try:
		count=0
		ind=1
		for i in (-np.array(b)).argsort()[:k]:
			if (i in (-np.array(a)).argsort()[:k]):
				count=count+1
				recall=recall+(count/float(k))
			ind=ind+1
		
		recall=recall/float(count)
	except:
		recall=0


	try:
		f1= 2* ((precision*recall)/(precision+recall))
	except:
		f1 = 0 
	return precision,recall,f1



def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def getMatrix():
	global matrix
	global bin_matrix
	return matrix,bin_matrix

def getScores():
	global scores
	return scores


def calc_ndcg(df, column,k):
    min_votes=5
    min_comments=30

    ndcg_global=[]


    grouped=df[df['tot'].astype(int)>min_votes].groupby('asin')

    for name, group in grouped:
        dffiltro = (df['asin']==name) & (df['tot'].astype(int)>min_votes)

        comments_count = df[dffiltro ]['tot'].values
        if ( (len(comments_count)>min_comments) ):

            values_test = df[dffiltro]['helpfulness'].values
            scores = df[dffiltro][column].values


            ind = (-np.array(scores)).argsort()
            a = np.array(values_test)[ind]	
            ndcg = ndcg_at_k(a, k)
            
            print "product="+str(name)+" ndcg="+str(ndcg)
            ndcg_global.append(ndcg)
    return ndcg_global




def executeFromDf(dfProducts, alpha=0.9, beta=-0.12, k=5):

	count=1
	corr_global=[]
	corr_word_global=[]
	min_votes=5
	min_comments=30
	#max_comments=100

	precision_global=[]
	recall_global=[]
	f1_global=[]

	ndcg_global=[]
	outputDataFrame=pd.DataFrame()

	grouped=dfProducts[dfProducts['tot'].astype(int)>min_votes].groupby('asin')

	for name, group in grouped:	
		dffiltro = (dfProducts['asin']==name) & (dfProducts['tot'].astype(int)>min_votes) 
		productDataFrame = pd.DataFrame(dfProducts[dffiltro])

		comments_count = dfProducts[dffiltro ]['tot'].values
		if ( (len(comments_count)>min_comments) ):
			count=count+1 
			clear_sentences=[]
			stars=[]

			word_count=[]
			for s in dfProducts[dffiltro].T.to_dict().values():
				word_count.append(len(s['reviewText'].split(" ")))
				clear_sentences.append(remove_stop(clear_string(s['reviewText'])))
				stars.append(float(s['overall']))


			scores= getMostSalientWithStar(clear_sentences,stars,10,alpha,beta)

			productDataFrame['powerWithStar']=scores
			outputDataFrame = pd.concat([outputDataFrame, productDataFrame])



			#########################################
			#############  METRICS  ################
			#########################################

			values_test = productDataFrame['helpfulness'].T.to_dict().values()
			

			ind = (-np.array(scores)).argsort()
			a = np.array(values_test)[ind]	

			ndcg = ndcg_at_k(a, k)
			ndcg_global.append(ndcg)


			print "product="+str(name) + " quant_documents=" +str(len(comments_count))  +  " ndcg_global="+str(np.mean(ndcg_global))+ " ndcg_local=" + str(ndcg) 
			if __name__ == "__main__":
				print "#"+str(count)+" product_id=" + str(name)
				print "total comentarios:" +str(len(clear_sentences))

				#print "precision="+str(np.mean(precision_global))
				print "ndcg="+str(np.mean(ndcg_global))+ " (" + str(ndcg) + ")"
				#print "recall="+str(np.mean(recall_global))
				#print "f1="+str(np.mean(f1_global))
				#print "corr word_count="+ str(corr_word_local)
				#print "corr word_count="+ str(np.mean(corr_word_global)) + " (" + str(corr_word_local) + ")"
				#print "correlacao local=" + str(corr_local)
				#print "corr MHR=" + str(np.mean(corr_global)) + " (" + str(corr_local) + ")"

				#print scores
				print "##################################"


	#

	return outputDataFrame,ndcg_global








##################
##### MAIN #######
##################
if __name__ == "__main__":

	#dataset='data/eletronic_sample.csv'
	dataset='data/eletronic_sample_counts.csv.gz'

	outputDataset='data/eletronic_scores.csv'


	dfProducts = pd.read_csv(dataset)


	#dfProducts['helpfulness']=dfProducts.apply(helpf,axis=1)
	#dfProducts['tot']=dfProducts.apply(tot,axis=1)

	dfProducts,ndcg_global= executeFromDf(dfProducts)

	#print calc_ndcg(dfProducts, 'powerWithStar',5)
	

	#dfProducts.to_csv(outputDataset)