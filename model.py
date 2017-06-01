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
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from fuzzywuzzy import fuzz
from gensim import models
#import spacy # Spacy uses GloVe word embeddings


from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


def load_data(test = False, extended = False):
   
    if not extended:
        train_df = pd.read_csv('data/train.csv', index_col = 'id', dtype = {'question1':str, 'question2':str})
        train_df.rename(columns = {'question1':'q1', 'question2':'q2'}, inplace = True)
        train_df['q1'] = train_df.apply(lambda x: x['q1'].lower(), axis = 1)
        train_df['q2'] = train_df.apply(lambda x: x['q2'].lower(), axis = 1)
        train_df.fillna(value = '')
        if test:
            test_df = pd.read_csv('data/test.csvt', index_col = 'test_id', dtype = {'question1':str, 'question2':str})
            test_df.rename(columns = {'question1':'q1', 'question2':'q2'}, inplace = True)
            test_df['q1'] = test_df.apply(lambda x: x['q1'].lower(), axis = 1)
            test_df['q2'] = test_df.apply(lambda x: x['q2'].lower(), axis = 1)
            test_df.fillna(value = '')
            return train_df, test_df
        else:
            return train_df
        
        
    elif extended:
        ext_keys = ['q1_token', 'q2_token','q1_stopwords', 'q2_stopwords', 'q1_wo_stopwords', 'q2_wo_stopwords']
        train_df = pd.read_csv('data/train_extended.csv', index_col = 'id', dtype = {'q1':str, 'q2':str})
        train_df.fillna(value = '')
        if test:
            test_df = pd.read_csv('data/test_extended.csv', index_col = 'test_id', dtype = {'q1':str, 'q2':str})
            test_df.fillna(value = '')
            for key in ext_keys:
                train_df[key] = train_df.apply(lambda x: ast.literal_eval(x[key]), axis = 1)
                test_df[key] = test_df.apply(lambda x: ast.literal_eval(x[key]), axis = 1 )
            return train_df, test_df
        else:
            for key in ext_keys:
                train_df[key] = train_df.apply(lambda x: ast.literal_eval(x[key]), axis = 1)
            return train_df

# TODO: create method for returning question list & freq
def create_question_list(train, test):
    questions_df = None
    
    
    
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

    
    tot_unique = list((set1 | set2))
    
    return tot_unique
    
def sep_unique(str1, str2):
    '''
    Return words unique to str1 (not in str2)
    '''
    set1 = set(str1)
    set2 = set(str2)
    
    unique = list(set1.difference(set2))
    
    return unique

def return_stopwords(str1, stops = None):
    '''
    Return stopwords froms tring
    '''
    if stops is None:
        stops = set(stopwords.words('english'))
    
    set1 = set(str1)
    stops = set(stopwords.words('english'))
    
    str_stopwords = list(set1.difference(stops))
    
    return str_stopwords

def remove_stopwords(str1, str2 = None, stops = None):
    '''
    Remove stopwords from string
    '''
    if stops is None:
        stops = set(stopwords.words('english'))
        
    if str2 == None:
        set1 = set(str1)
        
        str_wo_stopwords = list(set1.difference(stops))
    
    else:
        set1 = set(str1)
        set2 = set(str2)
        str_wo_stopwords = list(set1.difference(set2))
    
    return str_wo_stopwords

def return_specialchar(str1, tokenizer):
    '''
    Return special characters included in string
    '''
    if str1[-1] == '?':
        str1 = str1[:-1]
    
    spec_chars = tokenizer.tokenize(str1)
    spec_chars = [x.strip() for x in spec_chars if x != ' ']
    
    return spec_chars

def create_doc_vec(str1, bed, agg_type = 'avg'):
    '''
    Given a sting tokenization, aggregate words' embeddings with given agg_type
    '''
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

def wordvec_similarity(str1, str2, bed, agg_type = 'avg'):
    '''
    Calculate word embedding similarity
    '''
    doc_vec1 = create_doc_vec(str1, bed, agg_type = agg_type)
    doc_vec2 = create_doc_vec(str2, bed, agg_type = agg_type)
    
    similarity = sp.spatial.distance.cosine(doc_vec1, doc_vec2)
            
    return similarity
    
def wm_distance(str1, str2, bed):
    '''
    Calculate Word-Mover distance
    '''
    wm_dist = bed.wmdistance(str1, str2)
    
    return wm_dist
    
def extend_data(df, extended):
    '''
    Extended og dataframe by adding columns of various string subset tokenizations
    '''
    t_0 = time.time()
    stops = set(stopwords.words('english'))
    # Tokenize questions removing non alphanumeric characters
    tokenizer = RegexpTokenizer(r'\w+')
    df['q1_token'] = df.apply(lambda x: tokenizer.tokenize(x['q1']), axis = 1)
    df['q2_token'] = df.apply(lambda x: tokenizer.tokenize(x['q2']), axis = 1)
    t_1 = time.time()
    print('Questions Tokenized in {:.2f}s'.format(t_1-t_0))
    
    # Keep stopwords
    df['q1_stopwords'] = df.apply(lambda x: return_stopwords(x['q1_token'], stops = stops), axis = 1)
    df['q2_stopwords'] = df.apply(lambda x: return_stopwords(x['q2_token'], stops = stops), axis = 1)
    t_2 = time.time()
    print('Stopwords Retrieved in {:.2f}'.format(t_2- t_1))
    
    # Keep questions w/o stopwords
    df['q1_wo_stopwords'] = df.apply(lambda x: remove_stopwords(x['q1_token'], str2 = x['q1_stopwords']), axis = 1)
    df['q2_wo_stopwords'] = df.apply(lambda x: remove_stopwords(x['q2_token'], str2 = x['q2_stopwords']), axis = 1)
    t_3 = time.time()
    print('Stopwords Removed in {:.2f}'.format(t_3- t_2))
    
    # Keep special characters
    spec_char_tokenizer = RegexpTokenizer(r'\W+')
    df['q1_specchar'] = df.apply(lambda x: return_specialchar(x['q1'], spec_char_tokenizer), axis = 1)
    df['q2_specchar'] = df.apply(lambda x: return_specialchar(x['q2'], spec_char_tokenizer), axis = 1)
    t_4 = time.time()
    print('Special Characters Retrieved in {:.2f}'.format(t_4- t_3))
    
    df.to_csv('data/{}.csv'.format(extended))
    return df


def feature_gen(df, extended = None):
    '''
    Generate features given a dataframe of questions, tokens, etc.
    '''
    features = pd.DataFrame(index = df.index)
    
    if type(extended) == str:
        df = extend_data(df, extended)
        
    elif extended:
        pass
    
    t_0 = time.time()
        
    features['jaccard_full'] = df.apply(lambda x: jaccard(x['q1_token'], x['q2_token']), axis = 1)
    features['jaccard_wo_stop'] = df.apply(lambda x: jaccard(x['q1_wo_stopwords'], x['q2_wo_stopwords']), axis = 1) 
    features['jaccard_of_stop'] = df.apply(lambda x: jaccard(x['q1_stopwords'], x['q2_stopwords']), axis = 1)
    features['jaccard_specchar'] = df.apply(lambda x: jaccard(x['q1_specchar'], x['q2_specchar']), axis = 1)
    features['full_fuzz_ratio'] = df.apply(lambda x: fuzz.ratio(x['q1_token'], x['q2_token'])/100., axis = 1)
    t_1 = time.time()
    print('Jaccard-based features generated in {:.2f}s'.format(t_1-t_0))
    
    embedding_name = 'glove.6B.50d.w2v.txt'
    if '.bin' in embedding_name:
        binary = True
    else: 
        binary = False
    word_vec = models.KeyedVectors.load_word2vec_format('embeddings/{}'.format(embedding_name), binary = binary)
    t_2 = time.time()
    print('Word embeddings loaded in {:.2f}s'.format(t_2-t_1))
    features['full_similarity_avg'] = df.apply(lambda x: wordvec_similarity(x['q1_token'], x['q2_token'], word_vec, agg_type = 'avg'))
    #similarity_avg'] = 
    t_3 = time.time()
    print('Similarity-based efatures generated in {:.2f}s'.format(t_3-t_2))
    
    t_end = time.time()
    print('All features generated in {:.2f}'.format(t_end-t_0))
    
    '''
    embedding_name = 'glove.6B.50d.w2v.txt'
    if '.bin' in embedding_name:
        binary = True
    else: 
        binary = False
    word_vec = models.KeyedVectors.load_word2vec_format('embeddings/{}'.format(embedding_name), binary = binary)
    '''
    return features

def dump_classifier(clf, clf_name):
    '''
    Pickle classifier using clf_name
    '''
    if '.pickle' in clf_name:
        pass
    else:
        clf_name += '.pickle'
    with open('clfs/'+clf_name, 'wb') as f:
        pickle.dump(clf, f)
    f.close()
    
    print('Classifier dumped to clfs/{}'.format(clf_name))
    
    return

def NB_clf(features, labels):
    '''
    Train a Naive Bayes classifier. Return classifier, logloss scores & predictions
    '''
    X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size = 0.2)
    
    
    clf = GaussianNB()
    
    clf.fit(X_train, y_train)
    
    y_train_hat = clf.predict_proba(X_train)
    y_valid_hat = clf.predict_proba(X_valid)
    
    
    logloss_train = log_loss(y_train, y_train_hat)
    logloss_valid = log_loss(y_valid, y_valid_hat)
    
    clf_name = 'SVC_{0}_{1:.3f}'.format(time.strftime('%m-%d-%H'), logloss_valid)+'.pickle'
    dump_classifier(clf, clf_name)
    
    return clf, y_valid_hat, logloss_train, logloss_valid

def SVC_clf(features, labels, clf_pickle = None):
    '''
    Train a SVC classifier. Return classifier, logloss scores & predictions
    '''
    t_0 = time.time()
    X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size = 0.2)
    
    if clf_pickle == None:
        
        clf = SVC(kernel = 'rbf', probability = True)
        
    else:
        clf = clf_pickle
        
        
    clf.fit(X_train, y_train)
    t_1 = time.time()
    print('Classifier Fit in {:.2f}s'.format(t_1-t_0))

    y_train_hat = clf.predict_proba(X_train)
    y_valid_hat = clf.predict_proba(X_valid)
    
    
    logloss_train = log_loss(y_train, y_train_hat)
    logloss_valid = log_loss(y_valid, y_valid_hat)
    
    clf_name = 'SVC_{0}_{1:.3f}'.format(time.strftime('%m-%d-%H'), logloss_valid)+'.pickle'
    dump_classifier(clf, clf_name)
    
    return clf, y_valid_hat, logloss_train, logloss_valid

def xgb_clf(features, labels, clf_pickle = None):
    '''
    Train a xgb classifier. Return classifier, logloss scores & predictions
    '''
    print('Fittng XGBoost classifier...')
    t_0 = time.time()
    X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size = 0.2)
    
    if clf_pickle == None:
        clf = XGBClassifier()
    
    else:
        clf = clf_pickle
    
    clf.fit(X_train, y_train)
    t_1 = time.time()
    print('Classifier Fit in {:.2f}s'.format(t_1-t_0))
    
    y_train_hat = clf.predict_proba(X_train)
    y_valid_hat = clf.predict_proba(X_valid)
    
    
    logloss_train = log_loss(y_train, y_train_hat)
    logloss_valid = log_loss(y_valid, y_valid_hat)
    
    clf_name = 'SVC_{0}_{1:.3f}'.format(time.strftime('%m-%d-%H'), logloss_valid)+'.pickle'
    dump_classifier(clf, clf_name)
    
    return clf, y_valid_hat, logloss_train, logloss_valid

def main():
    extended = True
    # TODO: Change load_test to testing_mode (inverse)
    load_test = False
    if load_test:
        train_df, test_df = load_data(test = load_test, extended = extended)
        print('Train & Test Loaded')
    else:
        train_df = load_data(test = load_test, extended = extended)
        print('Train Loaded')
        
    if extended == True:
        features = feature_gen(train_df, extended = None)
        if load_test:
            features_test = feature_gen(test_df, extended = None)
    else:
        features = feature_gen(train_df, extended = 'train_extended')
        if load_test:
            features_test = feature_gen(test_df, extended = 'test_extended')
    
    features.to_csv('data/features.csv')
    if load_test:
        features_test.to_csv('data/features_test.csv')
        
    labels = train_df['is_duplicate']
    
    assert features.shape[0] == labels.shape[0]
    
    #clf, y_valid_hat, logloss_train, logloss_valid = NB_clf(features, labels)
    #clf, y_valid_hat, logloss_train, logloss_valid = SVC_clf(features, labels)
    clf, y_valid_hat, logloss_train, logloss_valid = xgb_clf(features, labels)
    
    print('Training Logloss: {}'.format(logloss_train))
    print("Validation Logloss: {}".format(logloss_valid))
    
    if load_test:
        # TODO: create prediction file
        pass
    
    
main()
    