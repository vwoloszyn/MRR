import sys, gzip, json, datetime
import nltk
import pandas as pd
import random


file_in="reviews_Electronics_5.json.gz"
file_out="data/eletronic_sample.csv"
min_sentences=3
min_votes = 5
min_comments=30
max_produtos=1000

g = gzip.open(file_in, 'r')


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



def extract_sentecens(text):

	return nltk.sent_tokenize(text)

products=[]
ind=0
count =0
for l in g:
    review_json = json.loads(l)

    if (len(review_json['reviewText'])>min_sentences):
    	products.append(review_json)

    ind=ind+1
    count=count+1
    if ind>1000:
    	print ("processado:+"+str(count))
    	ind=0



dfProducts=pd.DataFrame(products)

dfProducts['helpfulness']=dfProducts.apply(helpf,axis=1)
dfProducts['tot']=dfProducts.apply(tot,axis=1)


#min votes
dfProducts = dfProducts[dfProducts['tot'].astype(int)>min_votes]

#min comments
x=dfProducts.groupby('asin').size() >min_comments

mylist= list(x[x==True].keys())
sample = [ mylist[i] for i in sorted(random.sample(xrange(len(mylist)), max_produtos)) ]


#exporting
dfProducts[dfProducts['asin'].isin(sample)].to_csv(file_out)