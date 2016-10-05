import sys, gzip, json, datetime
import nltk
import pandas as pd
import random


n_produtos=500
file_in="reviews_Books_5.json.gz"
file_out="data/book_sample.csv.gz"
min_sentences=3
min_votes = 5
min_comments=30

g = gzip.open(file_in, 'r')


def helpf(x):
	x = str(x['helpful']) 
	try:
		pos = x.replace("[","").replace("]","").split(',')[0]
		neg = x.replace("[","").replace("]","").split(',')[1]
		tot = x.replace("[","").replace("]","").split(',')[1]
		return float ( float(pos) /  float(tot) )
	except:
		return 0

def tot(x): 
	x = str(x['helpful'])
	#print x.replace("[","").replace("]","").split(',')[1]
	try:
		return x.replace("[","").replace("]","").split(',')[1]
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
    if ind>10000:
    	print ("processado:+"+str(count))
    	ind=0

    if (count > 2000000):
    	break



dfProducts=pd.DataFrame(products)

dfProducts['helpfulness']=dfProducts.apply(helpf,axis=1)
dfProducts['tot']=dfProducts.apply(tot,axis=1)


#filtro de minimo de votos
dfProducts = dfProducts[dfProducts['tot'].astype(int)>min_votes]

#filtro minimo comentarios
x=dfProducts.groupby('asin').size() >min_comments

mylist= list(x[x==True].keys())

max_p=n_produtos
if (len(mylist)<n_produtos):
	max_p=len(mylist)
sample = [ mylist[i] for i in sorted(random.sample(xrange(len(mylist)), max_p)) ]



dfProducts[dfProducts['asin'].isin(sample)].to_csv(file_out, compression='gzip')