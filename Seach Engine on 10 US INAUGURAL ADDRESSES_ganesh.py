#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Required libraries
import os        
import nltk
#nltk.download('stopwords')
import math
from math import log10
from math import sqrt
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
from nltk.corpus import stopwords
stop_words = stopwords.words("english")
from nltk.stem.porter import PorterStemmer
porterstemmer = PorterStemmer()
corpusroot = 'C:\\Users\\laksh\\US_Inaugural_Addresses'

alldocs=[]  #List that contains all filenames with their respective terms
idf = {}      #Dictionary with idf values of the terms
tf_idf = {}    #Contains tf_idf weights of the tokens in the filename
lnc_wt={}       #Contains normalized weights of the document according to lnc weighting

for filename in os.listdir(corpusroot):
    if filename.startswith("0") or filename.startswith("1"): #Select filenames that start with 0 or 1
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        doc = file.read()
        file.close()
        doc = doc.lower()  #Set the words in each document to lower letters
        tokens = tokenizer.tokenize(doc)    # Tokenize the document
        tokens = [porterstemmer.stem(token) for token in tokens  if token not in stop_words] #Performing StopWord Removal and Stemming.
        alldocs.append((filename, tokens)) 
        
 # Function that returns the idf weight of a given token    
def getidf(tokn):
    for file_name, tokens in alldocs:
        for token in tokens:
            if token not in idf:
                df_term = len([tk for flnm, tk in alldocs  
                               if token in tk])           #calculates how many times a token appeared.
                idf[token] = log10(len(alldocs) / df_term) #len(alldocs)= 15 
    return idf.setdefault(tokn, -1)   #giving -1 as a default value to print, if token does not exist.     

# Function that returns the UnNormalized TF-IDF weights of a term in a filename.
def getweight(fname, tokn):
    stmmd= porterstemmer.stem(tokn)   #stemming token before calculation
    for fl_name, tokens in alldocs:
        tf_idf_vec = {}    #Holds values for tokens with their respective tf-idf scores
        doc_lnc_vec={}     #holds values according to "LNC WEIGHTING SCHEME"
        
        #Getting raw TF for both tfidf,and lnc weight calculations
        for token in tokens:
            tf_idf_vec[token] = tf_idf_vec.setdefault(token, 0) + 1   #used to get raw term frequencies of every token
            doc_lnc_vec[token] = doc_lnc_vec.setdefault(token, 0) + 1 #Getting raw TF for lnc_scheme
            
         #For computing TF-IDF Weights
        for token in tf_idf_vec: 
            logwtd_tf = 1+log10(tf_idf_vec[token])      #TF (logarithmic TF)
            tf_idf_vec[token] = logwtd_tf* idf[token]   #TF*IDF
            
        #Computing 'LNC' weighting Scheme[Document]    
        for token in doc_lnc_vec: #                                                     [LNC]
            logwtng=1+log10(doc_lnc_vec[token]) #using Logarithmic TF                    (L)
            doc_lnc_vec[token]=logwtng*1 #Using no IDF - multiply by (1)                 (N)
        sum_of_sqr=sum(val**2 for val in doc_lnc_vec.values())
        normalize = math.sqrt(sum_of_sqr)#Normalizing                                    
        
        for term, ln_wt in doc_lnc_vec.items():
            doc_lnc_vec[term] = ln_wt / normalize                                     #  (C) 
        lnc_wt[fl_name]= doc_lnc_vec
        tf_idf[fl_name] = tf_idf_vec
    if fname not in tf_idf:
        return 0
    return tf_idf[fname].setdefault(stmmd, 0)  

#Function to return tuple of document and score that is most similar to query acc to LNC.LTC
def query(querystr):
    tokens = tokenizer.tokenize(querystr.lower()) #PreProcess querystr :tokenize, remove stop words, and perform stemming
    tokens = [porterstemmer.stem(token) for token in tokens  if token not in stop_words]
    query_ltc_vec = {} #Vector for holding the 'ltc' weights(log wtd tf & idf & norm) of query string
    for token in tokens:
        if token in idf:
            query_ltc_vec[token] = query_ltc_vec.setdefault(token, 0) + 1 # To calculate raw TF of the query string!
            
    for token in query_ltc_vec: #Performing 'LTC' weighting scheme[Query]                       [LTC]
        logarithmic_tf= 1 + log10(query_ltc_vec[token]) #performing log TF                       (L)
        query_ltc_vec[token] = logarithmic_tf * idf[token] # TF-IDF Weights                      (T)
    sumsqr= sum(value**2 for value in query_ltc_vec.values())   
    normalized = math.sqrt(sumsqr)
    for term , tfidf_wt in query_ltc_vec.items():
        query_ltc_vec[term]= tfidf_wt/normalized                                    #            (C)
   
    similarity_values = {} #used to hold the values of cosine similarity scores
    for filename, doclnc_vec in lnc_wt.items(): #Perform cosine similarity for lnc,ltc vectors
        cosine_sim = sum(query_ltc_vec.get(tkn, 0) * doclnc_vec.get(tkn, 0) for tkn in set(query_ltc_vec) & set(doclnc_vec))
        similarity_values[filename] = cosine_sim
    best_match= max(similarity_values.items(), key=lambda top: top[1]) #Returns the document with highest query similarity
    max_score = max(similarity_values.values())
    if max_score == 0:
        return ("none", 0)
    else:
        return best_match 
    
    

print("%.12f" % getidf('british'))
print("%.12f" % getidf('union'))
print("%.12f" % getidf('war'))
print("%.12f" % getidf('power'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('02_washington_1793.txt','arrive'))
print("%.12f" % getweight('07_madison_1813.txt','war'))
print("%.12f" % getweight('12_jackson_1833.txt','union'))
print("%.12f" % getweight('09_monroe_1821.txt','great'))
print("%.12f" % getweight('05_jefferson_1805.txt','public'))
print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("false public"))
print("(%s, %.12f)" % query("people institutions"))
print("(%s, %.12f)" % query("violated willingly"))


# In[ ]:




