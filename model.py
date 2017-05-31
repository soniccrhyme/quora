#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 10:07:08 2017

testing word2vec, GloVe, spaCy, word embeddings

@author: ucalegon
"""

import pickle, time, ast
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss
from fuzzywuzzy import fuzz
from gensim import corpora, models, similarities
import spacy # Spacy uses GloVe word embeddings


from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


def load_data(test = False, extended = False):
    if not extended:
        if test:
            train_df = pd.read_csv('data/train.csv', index_col = 'id')
            test_df = pd.read_csv('data/test.csv', index_col = 'test_id')
            return train_df, test_df
        else:
            train_df = pd.read_csv('data/train.csv', index_col = 'id')
            return train_df
    elif extended:
        ext_keys = ['q1_token', 'q2_token','q1_stopwords', 'q2_stopwords', 'q1_wo_stopwords', 'q2_wo_stopwords']
        if test:
            train_df = pd.read_csv('data/train_extended.csv', index_col = 'id')
            test_df = pd.read_csv('data/test_extended.csv', index_col = 'test_id')
            for key in ext_keys:
                train_df[key] = train_df.apply(lambda x: ast.literal_eval(x[key]))
                test_df[key] = train_df.apply(lambda x: ast.literal_eval(x[key]))
            return train_df, test_df
        else:
            train_df = pd.read_csv('data/train_extended.csv', index_col = 'id')
            for key in ext_keys:
                train_df[key] = train_df.apply(lambda x: ast.literal_eval(x[key]))
            return train_df

def create_question_list(train, test):
    
    
    
    
    return questions_df
    
# TODO: finish train-test split func
def validation_split(df, y_name, test_size = 0.2):
    
    
    y = df[y_name]
    X = df.drop(y_name, axis = 1)
    
    
    X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size = test_size)
    
    return X_train, X_valid, y_train, y_valid
    

def jaccard(str1, str2):
    
    set1 = set(str1)
    set2 = set(str2)
    try:
        jaccard = float(len((set1 & set2)))/float(len((set1 | set2)))
    except ZeroDivisionError:
        jaccard = 0
    
    return jaccard
    

def common_words(str1, str2):
    
    set1 = set(str1)
    set2 = set(str2)

    common = (set1 & set2)

    return common

def all_unique(str1, str2):
    
    set1 = set(str1)
    set2 = set(str2)

    
    tot_unique = (set1 | set2)
    
    return tot_unique
    
def sep_unique(str1, str2):
    

    set1 = set(str1)
    set2 = set(str2)

    
    unique1 = set([w for w in set1 if w not in set2])
    unique2 = set([w for w in set2 if w not in set1])
    
    return unique1, unique2

def return_stopwords(str1):
    
    set1 = set(str1)
    stops = stopwords.words('english')
    
    str_stopwords = [w for w in set1 if w in stops]
    
    return str_stopwords

def remove_stopwords(str1):
    
    set1 = set(str1)
    stops = stopwords.words('english')
    
    str_wo_stopwords = [w for w in set1 if w not in stops]
    
    return str_wo_stopwords

def create_doc_vec(str1, agg_type = 'avg'):
    eg_vec = bed.word_vec('machine')
    
    M_1 = []
    for w in str1:
        try:
            w_vec = bed.word_vec(w)
        except:
            w_vec = np.zeros_like(eg_vec)
        M_1.append(w_vec)
        
    if agg_type == 'avg': 
        doc_vec = np.average(np.array(M_1), axis = 0)
        
    elif agg_type == 'min':
        doc_vec = np.min(np.array(M_1), axis = 0)
        
    elif agg_type == 'max':
        doc_vec = np.max(np.array(M_1), axis = 0)
    else:
        raise IOError('sim_type input must be in [avg, max, min]')
    
        
    return doc_vec

def wordvec_similarity(str1, str2, agg_type = 'avg', bed):

    doc_vec1 = create_doc_vec(str1, agg_type = agg_type)
    doc_vec2 = create_doc_vec(str2, agg_type = agg_type)
    
    similarity = sp.spatial.distance.cosine(doc_vec1, doc_vec2)
            
    return similarity
    
def wm_distance(str1, str2, bed):
    
    wm_dist = bed.wmdistance(str1, str2)
    
    return wm_dist
    
    


def feature_gen(df, extended = False):
    features = pd.DataFrame(index = df.index)
    
    if not extended:
        # Tokenize questions removing non alphanumeric characters
        tokenizer = RegexpTokenizer(r'\w+')
        df['q1_token'] = df.apply(lambda x: tokenizer.tokenize(x['q1']), axis = 1)
        df['q2_token'] = df.apply(lambda x: tokenizer.tokenize(x['q2']), axis = 1)
        print('Questions Tokenized')
        
        df['q1_stopwords'] = df.apply(lambda x: return_stopwords(x['q1_token']), axis = 1)
        df['q2_stopwords'] = df.apply(lambda x: return_stopwords(x['q2_token']), axis = 1)
        print('Stopwords Retrieved')
        
        df['q1_wo_stopwords'] = df.apply(lambda x: remove_stopwords(x['q1_token']), axis = 1)
        df['q2_wo_stopwords'] = df.apply(lambda x: remove_stopwords(x['q2_token']), axis = 1)
        print('Stopwords Removed')
    
    features['jaccard_full'] = df.apply(lambda x: jaccard(x['q1_token'], x['q2_token']), axis = 1)
    features['jaccard_wo_stop'] = df.apply(lambda x: jaccard(x['q1_wo_stopwords'], x['q2_wo_stopwords']), axis = 1) 
    features['jaccard_of_stop'] = df.apply(lambda x: jaccard(x['q1_stopwords'], x['q2_stopwords']), axis = 1)
    features['full_fuzz_ratio'] = df.apply(lambda x: fuzz.ratio(x['q1'], x['q2'])/100., axis = 1)
    
    return features, df

def train_predict(features, labels):
    
    X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size = 0.2)
    
    
    clf = GaussianNB()
    
    clf.fit(X_train, y_train)
    
    y_valid_hat = clf.predict_proba(X_valid)
    
    logloss = log_loss(y_valid, y_valid_hat)
    
    return clf, y_valid_hat, logloss



def main():
    train_df, test_df = load_data(test = True)
    
    train_df.rename(columns = {'question1':'q1', 'question2':'q2'}, inplace = True)
    test_df.rename(columns = {'question1':'q1', 'question2':'q2'}, inplace = True)
    train_df['q1'] = train_df['q1'].astype(str)
    train_df['q1'] = train_df.apply(lambda x: x['q1'].lower(), axis = 1)
    train_df['q2'] = train_df['q2'].astype(str)
    train_df['q2'] = train_df.apply(lambda x: x['q2'].lower(), axis = 1)
    test_df['q1'] = test_df['q1'].astype(str)
    test_df['q1'] = test_df.apply(lambda x: x['q1'].lower(), axis = 1)
    test_df['q2'] = test_df['q2'].astype(str)
    test_df['q2'] = test_df.apply(lambda x: x['q2'].lower(), axis = 1)
    
    features, train_df_exp = feature_gen(train_df)
    print('Features generated')
    labels = train_df_exp['is_duplicate']
    clf, y_valid_hat, logloss = train_predict(features, labels)
    
    
    '''
    embedding_name = 'glove.6B.50d.w2v.txt'
    if '.bin' in embedding_name:
        binary = True
    else: 
        binary = False
    
    global word_vec
    word_vec = models.KeyedVectors.load_word2vec_format('embeddings/{}'.format(embedding_name), binary = binary)
    '''
    
    print("Validation Logloss: {}".format(logloss))
    with open('clfs/NB_1.pickle', 'rw') as f:
        pickle.load(f, clf)
    f.close()
    
    
main()
    