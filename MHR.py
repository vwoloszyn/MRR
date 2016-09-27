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


dataset='data/eletronic_sample.csv'
outputDataset='data/eletronic_scores.csv'


dfProducts = pd.read_csv(dataset)

#filtrar somente o produto com a maior quantidade de comentarios
##x=dfProducts.groupby('asin').size().sort_values(ascending=0)[:2]
##mylist= list(x.keys())
##dfProducts = dfProducts[dfProducts['asin'].isin(mylist)]

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
		value = re.sub(w, "", value)
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

    def __init__(self, beta=0.85, epsilon=0.0001):
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
	return matrix_sentences
    


def create_matrix_stars(stars):
	matrix_stars=np.zeros((len(stars), len(stars)))

	for row in range(len(stars)):
		for col in range(len(stars)):
			matrix_stars[row][col]= abs(stars[row]-stars[col])+1
			#print "ditance" + str(matrix_stars[row][col])


	matrix_stars = matrix_stars/5

	return matrix_stars



def getMostSalientWithStar(comments, stars, comments_count):

	threshold = 0.01
	epsilon = 0.01

	alpha=1


	matrix_sentences = create_matrix_sentence(comments )
	#matrix_sentences = compute_distance_concept(comments)

	matrix_stars = create_matrix_stars(stars)

	

	matrix = np.array(alpha*matrix_sentences) / np.array(matrix_stars)


	threshold = np.mean(matrix_sentences)*1.7
	for row in range(len(matrix)):
		for col in range(len(matrix)):
			if matrix[row, col] > threshold:
			    matrix[row, col] = 1.0
			    #degrees[row] += 1
			else:
			    matrix[row, col] = 0

	
	#print matrix
	pr1 = PageRank_NoTeleport()
	scores = pr1.compute(matrix)
	
	#print scores
	
	return scores


def cal_metrics(a,b,n_ref,n_feat):
	precision=0
	recall=0
	f1=0
	#print b
	try:
		count=0
		ind=1
		for i in (-np.array(b)).argsort()[:n_feat]:
			if (i in (-np.array(a)).argsort()[:n_ref]):
				count=count+1
				precision=precision+(count/float(ind))
			ind=ind+1
		
		precision=precision/float(count)
	except:
		precision=0
	#precision=len(np.intersect1d(-(np.array(a)).argsort()[:n_ref], -(np.array(b)).argsort()[:n_feat] ))/float(n_feat)

	#recall=len(np.intersect1d(-(np.array(a)).argsort()[:n_ref], -(np.array(b)).argsort()[:n_feat] ))/float(n_ref)

	try:
		count=0
		ind=1
		for i in (-np.array(b)).argsort()[:n_feat]:
			if (i in (-np.array(a)).argsort()[:n_ref]):
				count=count+1
				recall=recall+(count/float(n_ref))
			ind=ind+1
		
		recall=recall/float(count)
	except:
		recall=0


	try:
		f1= 2* ((precision*recall)/(precision+recall))
	except:
		f1 = 0 
	return precision,recall,f1







##################
##### MAIN #######
##################




dfProducts['helpfulness']=dfProducts.apply(helpf,axis=1)
dfProducts['tot']=dfProducts.apply(tot,axis=1)


count=1
corr_global=[]
corr_word_global=[]
min_votes=5
min_comments=40
#max_comments=100

precision_global=[]
recall_global=[]
f1_global=[]

grouped=dfProducts[dfProducts['tot'].astype(int)>min_votes].groupby('asin')

for name, group in grouped:	
	dffiltro = (dfProducts['asin']==name) & (dfProducts['tot'].astype(int)>min_votes) 

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


		scores= getMostSalientWithStar(clear_sentences,stars,10)

		dfProducts.loc[dffiltro,'powerWithStar']=scores



		#########################################
		#############  METRICS  ################
		#########################################

		values_test = dfProducts[dffiltro]['helpfulness'].T.to_dict().values()
		corr_local=np.corrcoef(values_test,scores)[0][1]
		corr_word_local=np.corrcoef(values_test,word_count)[0][1]

		corr_word_global.append(corr_word_local)
		corr_global.append(corr_local)


		precision, recall, f1 = cal_metrics(values_test,scores,5,10)
		precision_global.append(precision)
		recall_global.append(recall)
		f1_global.append(f1)
		
		print "#"+str(count)+" product_id=" + str(name)
		print "total comentarios:" +str(len(clear_sentences))

		#print "precision="+str(np.mean(precision_global))
		#print "recall="+str(np.mean(recall_global))
		#print "f1="+str(np.mean(f1_global))
		#print "corr word_count="+ str(corr_word_local)
		print "corr word_count="+ str(np.mean(corr_word_global)) + " (" + str(corr_word_local) + ")"
		#print "correlacao local=" + str(corr_local)
		print "corr MHR=" + str(np.mean(corr_global)) + " (" + str(corr_local) + ")"

		#print scores
		print "##################################"


dfProducts.to_csv(outputDataset)
