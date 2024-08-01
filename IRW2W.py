
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 02:19:13 2020

@author: Manjusha Pattadkal
"""

import pandas as pd
#import sys
#import tokenize
import numpy as np
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
#from nltk.tokenize import  porter
#nltk.download()
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from gensim.models import word2vec
from gensim.models import KeyedVectors



#-----------------------------------------------------------------------------------------------------
#Importing Data

datapoints = pd.read_csv(r"C:\Users\Manjusha Pattadkal\Downloads\quora-question-pairs\train.csv\train.csv",engine='python')
datapoints.dropna(axis=0,inplace=True)
#datalist = datapoints.values.tolist()
datapoints.var(ddof=0,axis=1)
rowcount = len(datapoints.axes[0])

#------------------------------------------------------------------------------------------------------
#Extracting useful columns
indexlist = []
train = round (0.75 * rowcount)


datapoints.dtypes

# extract information of datatypes of the columns
filteredColumns = datapoints.dtypes[datapoints.dtypes == np.object]
# get object datatype columns
indexlist = list(filteredColumns.index)


#numberlist = []
#
#
#for i in range  (0,len(indexlist)):
#    numberlist.append(0)
#    numberlist[i] = datapoints.columns.get_loc(indexlist[i])
#numberlist.append(0)
# 
#print (len(numberlist))

questions1 = []
questions2 = []

questions1 = datapoints[indexlist[0]].values.tolist()
questions2 = datapoints[indexlist[1]].values.tolist()
output = datapoints.iloc[:,-1].values.tolist()

questions1 = questions1[train:]
questions2 = questions2[train:]

train = rowcount-train

#-----------------------------------------------------------------------------------------------------



def preprocessing(String, stopwordFlag=True , stemmingFlag=True): #default value is always true for stemming and stopwords
    
    '''
    This function is used for preprocessing
    - Tokenization
    - Stemming
    - Stop Words
        
    '''
    token = nltk.word_tokenize(String)
    if stopwordFlag==False and stemmingFlag == False:
         token_string = " ".join(token)
         return token_string
    
    stop_words = set(stopwords.words('english'))
    influentialwords = []
    stemwords = []
    for w in token:
        if w not in stop_words: 
            influentialwords.append(w)
    if stemmingFlag == False:
        stopwords_string = " ".join(stemwords)
        return stopwords_string
        
    ps= PorterStemmer()
    
    if stopwordFlag==False:
        for w in token:
            stemwords.append(ps.stem(w))
        stemwords_string = " ".join(stemwords)
        return stemwords_string
    for w in influentialwords:
         stemwords.append(ps.stem(w))
    stemwords_string = " ".join(stemwords)
    return stemwords

#------------------------------------------------------------------------------------------

# def vectorization(questions):
#     '''
    

#     Parameters
#     ----------
#     questions : TYPE String
#         DESCRIPTION. questions list

#     Returns
#     -------
#     vectormatrix : TYPE dense matrix
#         DESCRIPTION. tf-idf vector

#     '''
#     tfidf = []
#     tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
#     quetokens_df = pd.DataFrame(questions)
#     tfidf = tfidf_vectorizer.fit_transform(quetokens_df[0])
#     vectormatrix = tfidf.todense()
#     return vectormatrix
#----------------------------------------------------------------------------------------------



def Similarity(question1,question2):
    score = cosine_similarity(question1,question2)
    return score
#----------------------------------------------------------------------------------------------------
def evaluate(Similarity,target,threshold,train):
    count = 0 
    predicted = []
    for i in range (0,train):
        if Similarity[i] > threshold:
            predicted.append(1)
        else:
            predicted.append(0)
    # for j in range (0,train):
    #     if predicted[j]==target[j]:
    #         #print("Yes")
    #         count=count+1
    # print(predicted[:6],target[:6])
    # print(count)
    # accuracy = float(count/train)
#    print (predicted,target[:train])
    accuracy = accuracy_score(target[:train],predicted)
    return accuracy
#----------------------------------------------------------------------------------------------------


#main
que1tokens = []
que2tokens = []
similarityscore = []
for i in range (0,train):
    que1tokens.append(preprocessing(questions1[i]))
    que2tokens.append(preprocessing(questions2[i]))

for i in range (0,train):
    combining = que1tokens.append(que2tokens[i])
    
# VectorQue1 = vectorization(que1tokens)
# #VectorQue2 = vectorization(que2tokens)

# for i in range (0,train):
#     similarityscore.append(Similarity(VectorQue1[train+i,:],VectorQue1[i,:] ))


# print(evaluate(similarityscore,output[:train],0.99819,train))

model = KeyedVectors.load_word2vec_format(r'C:\Users\Manjusha Pattadkal\Documents\GoogleNews-vectors-negative300.bin.gz', binary=True)


#model = word2vec.Word2Vec(que1tokens, workers = 10, size = 300, min_count = 1, window = 3, sg = 0)
similar_word = model.wv.most_similar('good')[0]


def avg_sentence(sentence, wv):
  v = np.zeros(300)
  if len(sentence) == 0:
      length = 1
  else:
      length = len(sentence)
  for w in sentence:
    if w in wv:
      v += wv[w]
  return v / length


for i in range(0,train):
    que1 = avg_sentence(que1tokens[i],model.wv).reshape(1,300)
    que2 = avg_sentence (que1tokens[train+i],model.wv).reshape(1,300)
#    print(que1,que2)
    similarityscore.append(Similarity(que1,que2))
    
    
    
print(evaluate(similarityscore,output[:train],0.89,train))
           




# import gzip
# from gensim.models import KeyedVectors

# f = gzip.open(r'C:\Users\Manjusha Pattadkal\Documents\GoogleNews-vectors-negative300.bin.gz', 'rb')

# model = KeyedVectors.load_word2vec_format(r'C:\Users\Manjusha Pattadkal\Documents\GoogleNews-vectors-negative300.bin.gz', binary=True)

  


#0.89 --- 66%  59%




