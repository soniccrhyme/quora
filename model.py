#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 10:07:08 2017

testing word2vec, GloVe, spaCy, word embeddings

@author: ucalegon
"""

import pickle, time, ast
import numpy as np
import scipy
import pandas as pd
from fuzzywuzzy import fuzz
#from gensim import models
import spacy # Spacy uses GloVe word embeddings
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


train_size = 404290
test_size = 2345796




def load_data(test = False):
    '''
    Load data from different csv files depending on mode.
    Extended already contains the fields generated in extend_data()
    '''
        
    ext_keys = ['q1_token', 'q2_token','q1_stopwords', 'q2_stopwords', 'q1_wo_stopwords', 'q2_wo_stopwords']
    train_df = pd.read_csv('data/train_extended.csv', index_col = 'id', dtype = {'q1':str, 'q2':str})
    train_df.fillna(value = '', inplace = True)
    if test:
        test_df = pd.read_csv('data/test_extended.csv', index_col = 'test_id', dtype = {'q1':str, 'q2':str})
        test_df.fillna(value = '', inplace = True)
        for key in ext_keys:
            #train_df[key] = [ast.literal_eval(x) for x in train_df[key].tolist()]
            train_df[key] = train_df.apply(lambda x: ast.literal_eval(x[key]), axis = 1)
            #test_df[key] = [ast.literal_eval(x) for x in test_df[key].tolist()]
            test_df[key] = test_df.apply(lambda x: ast.literal_eval(x[key]), axis = 1)
        return train_df, test_df
    else:
        for key in ext_keys:
            train_df[key] = train_df.apply(lambda x: ast.literal_eval(x[key]), axis = 1)
        return train_df

def load_data_from_pickle(test = False):
    train_df = pd.read_pickle('data/train_extended.pickle')
    if test:
        test_df = pd.read_pickle('data/test_extended.pickle')
        return train_df, test_df
    else:
        return train_df

def jaccard(str1, str2):
    '''
    Return Jaccard score (|common|/|unique|) for questions
    '''
    set1 = set(str1)
    set2 = set(str2)
    try:
        jaccard = float(len((set1 & set2)))/float(len((set1 | set2)))
    except ZeroDivisionError:
        jaccard = 0
    
    return jaccard

def jaccard_list(list1, list2):
    out = []
    
    assert len(list1) == len(list2)
    for i in range(len(list1)):
        set1 = set(list1[i])
        set2 = set(list2[i])
        try:
            jaccard = float(len((set1 & set2)))/float(len((set1 | set2)))
        except ZeroDivisionError:
            jaccard = 0
        out.append(jaccard)
    return out
        
        
    
    

def common_words(str1, str2):
    '''
    Return words common to both questions
    '''
    
    set1 = set(str1)
    set2 = set(str2)

    common = (set1 & set2)

    return common

def all_unique(str1, str2):
    '''
    Return all unique words used by both questions
    '''
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

def word_share(str1, str2):
    
    if len(str1) == 0:
        return 0
    else:
        word_share = len([s1 for s1 in str1 if s1 in str2])/len(str1)
        return word_share

def create_doc_matrix(str1, bed_dict, idfs = None):
    '''
    Given a sting tokenization, aggregate words' embeddings with given agg_type
    '''
    if idfs is None:
        M = []
        for s in str1:
            if s in bed_dict.keys():
                M.append(np.array(bed_dict[s], dtype = np.float32).reshape((300)))
            else:
                continue
        if len(M) == 0:
            M = [np.zeros((1,300))]
        return np.array(M, dtype = np.float32)
    else:
        M = []
        for s in str1:
            if (s in idfs.keys()) & (s in bed_dict.keys()):
                M.append(np.array(bed_dict[s], dtype = np.float32).reshape((300)))
            else:
                M.append(np.zeros((300)))
        if len(M) == 0:
            M = [np.zeros((1,300))]
        return np.array(M, dtype = np.float32)


def vec_similarity(str1, str2, bed_dict, agg_type = 'avg', sim_type = 'cosine'):
    '''
    Calculate word embedding similarity
    '''
    M_s1 = create_doc_matrix(str1, bed_dict)
    M_s2 = create_doc_matrix(str2, bed_dict)
    
    if agg_type == 'avg':
        s1_vec = np.average(np.array(M_s1), axis = 0)
        s2_vec = np.average(np.array(M_s2), axis = 0)
    
    elif agg_type == 'min':
        s1_vec = np.min(np.array(M_s1), axis = 0)
        s2_vec = np.min(np.array(M_s2), axis = 0)
        
    elif agg_type == 'max':
        s1_vec = np.max(np.array(M_s1), axis = 0)
        s2_vec = np.max(np.array(M_s2), axis = 0)


    if sim_type == 'cosine': 
        zeros = [np.zeros((1,300))]
        if np.all(s1_vec == zeros) | np.all(s2_vec == zeros):
            return 0
        else:
            similarity = 1-scipy.spatial.distance.cosine(s1_vec, s2_vec)
            return similarity
    
    elif sim_type == 'euclidean':
        similarity = scipy.spatial.distance.euclidean(s1_vec, s2_vec)
        return similarity
    
    elif sim_type == 'manhattan':
        similarity = scipy.spatial.distance.cityblock(s1_vec, s2_vec)
        return similarity
    
    elif sim_type == 'correlation':
        similarity = scipy.spatial.distance.correlation(s1_vec, s2_vec)
        if np.isnan(similarity):
            return 0
        else:
            return similarity
        
        
def weighted_vec_similarity(str1, str2, bed_dict, idfs, sim_type = 'cosine'):
    '''
    Calculate word embedding similarity weighted by idf
    '''
    M_s1 = create_doc_matrix(str1, bed_dict, idfs = idfs)
    M_s2 = create_doc_matrix(str2, bed_dict, idfs = idfs)
    
    eg = bed_dict['the']
    
    s1_weights = []
    for s in str1:
        if (s in idfs.keys()) & (s in bed_dict.keys()):
            s1_weights.append(idfs[s])
        else:
            s1_weights.append(0)
    s2_weights = []
    for s in str2:
        if (s in idfs.keys()) & (s in bed_dict.keys()):
            s2_weights.append(idfs[s])
        else:
            s2_weights.append(0)
    if np.sum(s1_weights) == 0:
        s1_vec = np.zeros_like(eg)
    else:
        s1_vec = np.average(np.array(M_s1), weights = s1_weights, axis = 0)
    if np.sum(s2_weights) == 0:
        s2_vec = np.zeros_like(eg)
    else:
        s2_vec = np.average(np.array(M_s2), weights = s2_weights, axis = 0)
    
    
    
    
    
    if sim_type == 'cosine': 
        zeros = [np.zeros((1,300))]
        if np.all(s1_vec == zeros) | np.all(s2_vec == zeros):
            return 0
        else:
            similarity = 1-scipy.spatial.distance.cosine(s1_vec, s2_vec)
            return similarity
    
    elif sim_type == 'euclidean':
        similarity = scipy.spatial.distance.euclidean(s1_vec, s2_vec)
        return similarity
    
    elif sim_type == 'manhattan':
        similarity = scipy.spatial.distance.cityblock(s1_vec, s2_vec)
        return similarity
    
    elif sim_type == 'correlation':
        similarity = scipy.spatial.distance.correlation(s1_vec, s2_vec)
        if np.isnan(similarity):
            return 0
        else:
            return similarity
    
    
def weighted_docvec(str1, str2, bed_dict, idfs):
    '''
    Calculate word embedding similarity weighted by idf
    '''
    M_s1 = create_doc_matrix(str1, bed_dict, idfs = idfs)
    M_s2 = create_doc_matrix(str2, bed_dict, idfs = idfs)
    
    eg = bed_dict['the']
    
    s1_weights = []
    for s in str1:
        if (s in idfs.keys()) & (s in bed_dict.keys()):
            s1_weights.append(idfs[s])
        else:
            s1_weights.append(0)
    s2_weights = []
    for s in str2:
        if (s in idfs.keys()) & (s in bed_dict.keys()):
            s2_weights.append(idfs[s])
        else:
            s2_weights.append(0)
    if np.sum(s1_weights) == 0:
        s1_vec = np.zeros_like(eg)
    else:
        s1_vec = np.average(np.array(M_s1), weights = s1_weights, axis = 0)
    if np.sum(s2_weights) == 0:
        s2_vec = np.zeros_like(eg)
    else:
        s2_vec = np.average(np.array(M_s2), weights = s2_weights, axis = 0)
        
    out = np.append(s1_vec, s2_vec, axis = 0)
        
    return out
        

'''
#DEPRECATED
def wm_distance(str1, str2):
    wm_dist = word_vec.wmdistance(str1, str2)
    
    return wm_dist
    
'''
    
# TODO: Finish normalize func
def normalize(features, labels = None):
    if labels == None:
        labels = list(features.columns)
        
    for key in labels:
        features[key] = scipy.stats.mstats.zscore(features[key])
        
    return features


def feature_gen(df, embedding_dict):
    '''
    Generate features given a dataframe of questions, tokens, etc.
    '''
    features = pd.DataFrame(index = df.index)
    
    t_0 = time.time()
        
    # Jaccard features
    features['jaccard_full'] = jaccard_list(df.q1_token.tolist(), df.q2_token.tolist())
    features['jaccard_wo_stop'] = jaccard_list(df.q1_wo_stopwords.tolist(), df.q2_wo_stopwords.tolist())
    features['jaccard_of_stop'] = jaccard_list(df.q1_stopwords.tolist(), df.q2_stopwords.tolist())
    features['jaccard_specchar'] = jaccard_list(df.q1_specchar.tolist(), df.q2_specchar.tolist())
    t_1 = time.time()
    print('Jaccard-based features generated in {:.2f}s @ {}'.format(t_1-t_0, time.strftime('%H:%M', time.localtime())))
    print(features.shape)
    assert features.shape[0] == df.shape[0]
    
    # Fuzzywuzzy features
    features['fuzz_QRatio_full'] = [fuzz.QRatio(x, y)/100. for x,y in zip(df.q1.tolist(), df.q2.tolist())]
    features['fuzz_QRatio_tks'] = [fuzz.QRatio(x, y)/100. for x,y in zip(df.q1_token.tolist(), df.q2_token.tolist())]
    features['fuzz_WRatio_full'] = [fuzz.WRatio(x, y)/100. for x,y in zip(df.q1.tolist(), df.q2.tolist())]
    features['fuzz_WRatio_tks'] = [fuzz.WRatio(x, y)/100. for x,y in zip(df.q1_token.tolist(), df.q2_token.tolist())]
    features['fuzz_partial_full'] = [fuzz.partial_ratio(x, y)/100. for x,y in zip(df.q1.tolist(), df.q2.tolist())] 
    features['fuzz_partial_tks'] = [fuzz.partial_ratio(' '.join(x), ' '.join(y))/100. for x,y in zip(df.q1_token.tolist(), df.q2_token.tolist())] 
    features['fuzz_partknset_full'] = [fuzz.partial_token_set_ratio(x, y)/100. for x,y in zip(df.q1.tolist(), df.q2.tolist())]
    features['fuzz_partknset_tks'] = [fuzz.partial_token_set_ratio(' '.join(x), ' '.join(y))/100. for x,y in zip(df.q1_token.tolist(), df.q2_token.tolist())]
    features['fuzz_partknsort_full'] = [fuzz.partial_token_sort_ratio(x, y)/100. for x,y in zip(df.q1.tolist(), df.q2.tolist())]
    features['fuzz_partknsort_tks']  = [fuzz.partial_token_sort_ratio(' '.join(x), ' '.join(y))/100. for x,y in zip(df.q1_token.tolist(), df.q2_token.tolist())]
    
    features['fuzz_tknset_full'] = [fuzz.token_set_ratio(x, y)/100. for x,y in zip(df.q1.tolist(), df.q2.tolist())]
    features['fuzz_tknset_tks'] = [fuzz.token_set_ratio(' '.join(x), ' '.join(y))/100. for x,y in zip(df.q1_token.tolist(), df.q2_token.tolist())]
    features['fuzz_tknsort_full'] = [fuzz.token_sort_ratio(x, y)/100. for x,y in zip(df.q1.tolist(), df.q2.tolist())]
    features['fuzz_tknsort_tks'] = [fuzz.token_sort_ratio(' '.join(x), ' '.join(y))/100. for x,y in zip(df.q1_token.tolist(), df.q2_token.tolist())]
    t_1_2 = time.time()
    print('Fuzzywuzzy based features generated in {:.2f} @ {}'.format(t_1_2-t_1, time.strftime('%H:%M', time.localtime())))
    print(features.shape)
    #assert features.shape[0] == df.shape[0]
    
    
    nlp = spacy.load('en')
    
    
    # Embedding similarity features, via Spacy
    features['similarity_spacy_full'] = [nlp(' '.join(x)).similarity(nlp(' '.join(y))) for x,y in zip(df.q1_token.tolist(), df.q2_token.tolist())]
    features['similarity_spacy_unique'] = [nlp(' '.join(sep_unique(x, y))).similarity(nlp(' '.join(sep_unique(y, x)))) for x,y in zip(df.q1_token.tolist(), df.q2_token.tolist())]
    
    
    features['sim_min_cosine'] = [vec_similarity(x,y, embedding_dict, agg_type = 'min') for x,y in zip(df.q1_wo_stopwords.tolist(), df.q2_wo_stopwords.tolist())]
    features['sim_max_cosine'] = [vec_similarity(x,y, embedding_dict, agg_type = 'max') for x,y in zip(df.q1_wo_stopwords.tolist(), df.q2_wo_stopwords.tolist())]
    
    features['sim_avg_euclid'] = [vec_similarity(x,y, embedding_dict, agg_type = 'avg', sim_type = 'euclidean') for x,y in zip(df.q1_wo_stopwords.tolist(), df.q2_wo_stopwords.tolist())]
    features['sim_max_euclid'] = [vec_similarity(x,y, embedding_dict, agg_type = 'max', sim_type = 'euclidean') for x,y in zip(df.q1_wo_stopwords.tolist(), df.q2_wo_stopwords.tolist())]
    features['sim_min_euclid'] = [vec_similarity(x,y, embedding_dict, agg_type = 'min', sim_type = 'euclidean') for x,y in zip(df.q1_wo_stopwords.tolist(), df.q2_wo_stopwords.tolist())]
    
    
    features['sim_avg_manhattan'] = [vec_similarity(x,y, embedding_dict, agg_type = 'avg', sim_type = 'manhattan') for x,y in zip(df.q1_wo_stopwords.tolist(), df.q2_wo_stopwords.tolist())]
    features['sim_max_manhattan'] = [vec_similarity(x,y, embedding_dict, agg_type = 'max', sim_type = 'manhattan') for x,y in zip(df.q1_wo_stopwords.tolist(), df.q2_wo_stopwords.tolist())]
    features['sim_min_manhattan'] = [vec_similarity(x,y, embedding_dict, agg_type = 'min', sim_type = 'manhattan') for x,y in zip(df.q1_wo_stopwords.tolist(), df.q2_wo_stopwords.tolist())]
    
    features['sim_avg_correlation'] = [vec_similarity(x,y, embedding_dict, agg_type = 'avg', sim_type = 'correlation') for x,y in zip(df.q1_wo_stopwords.tolist(), df.q2_wo_stopwords.tolist())]
    features['sim_max_correlation'] = [vec_similarity(x,y, embedding_dict, agg_type = 'max', sim_type = 'correlation') for x,y in zip(df.q1_wo_stopwords.tolist(), df.q2_wo_stopwords.tolist())]
    features['sim_min_correlation'] = [vec_similarity(x,y, embedding_dict, agg_type = 'min', sim_type = 'correlation') for x,y in zip(df.q1_wo_stopwords.tolist(), df.q2_wo_stopwords.tolist())]
    #assert features.shape[0] == df.shape[0]
    # Load idfs for weighted_averages
    with open('data/idf_weights.pickle', 'rb') as f:
        idfs = pickle.load(f)
    f.close()
    
    
    features['sim_weighted_avg_cosine'] = [weighted_vec_similarity(x,y, embedding_dict, idfs, sim_type = 'cosine') for x,y in zip(df.q1_wo_stopwords.tolist(), df.q2_wo_stopwords.tolist())]
    features['sim_weighted_avg_euclid'] = [weighted_vec_similarity(x,y, embedding_dict, idfs, sim_type = 'euclidean') for x,y in zip(df.q1_wo_stopwords.tolist(), df.q2_wo_stopwords.tolist())]
    features['sim_weighted_avg_manhattan'] = [weighted_vec_similarity(x,y, embedding_dict, idfs, sim_type = 'manhattan') for x,y in zip(df.q1_wo_stopwords.tolist(), df.q2_wo_stopwords.tolist())]
    features['sim_weighted_avg_correlation'] = [weighted_vec_similarity(x,y, embedding_dict, idfs, sim_type = 'correlation') for x,y in zip(df.q1_wo_stopwords.tolist(), df.q2_wo_stopwords.tolist())]
    
    
    t_2 = time.time()
    print('Similarity features generated in {:.2f}s @ {}'.format(t_2-t_1_2, time.strftime('%H:%M', time.localtime())))
    print(features.shape)
    #assert features.shape[0] == df.shape[0]
    
    
    
    # Question count (leaky) features, from question_counts.py output
    if (features.shape[0] == train_size) or (features.shape[0] == 25000):
        q_count = pd.read_csv('data/train_checked_q_count.csv', index_col = 'id')
    elif features.shape[0] == test_size:
        q_count = pd.read_csv('data/test_checked_q_count.csv', index_col = 'test_id')
    features['q1_freq'] = q_count['q1_freq']
    features['q2_freq'] = q_count['q2_freq']
    features['q1_q2_intersect'] = q_count['q1_q2_intersect']
    del q_count
    t_5 = time.time()
    print('Question frequency features added in {:.2f}s @ {}'.format(t_5-t_2, time.strftime('%H:%M', time.localtime())))
    print(features.shape)
    #assert features.shape[0] == df.shape[0]
    
    
    # Word-Share Features
    features['q1_in_q2'] = [word_share(x,y) for x,y in zip(df.q1_token.tolist(), df.q2_token.tolist())]
    features['q2_in_q1'] = [word_share(x,y) for x,y in zip(df.q2_token.tolist(), df.q1_token.tolist())]
    
    # Generic count features
    features['q1_word_count'] = [len(x) for x in df.q1_token.tolist()]
    features['q2_word_count'] = [len(x) for x in df.q2_token.tolist()]
    
    
    features['q1_char_len'] = [len(x.replace(' ', '')) for x in df.q1.tolist()]
    features['q2_char_len'] = [len(x.replace(' ', '')) for x in df.q2.tolist()]
    
    features['q1_len'] = [len(x) for x in df.q1.tolist()]
    features['q2_len'] = [len(x) for x in df.q2.tolist()]
    
    t_end = time.time()
    print('All features generated in {:.2f} @ {}; {} rows'.format(t_end-t_0, time.strftime('%H:%M', time.localtime()), features.shape[0]))
    print(features.shape)
    #assert features.shape[0] == df.shape[0]
    
    
    return features

def docvec_feature_gen(df, embedding_dict):
    
    
    
    with open('data/idf_weights.pickle', 'rb') as f:
        idfs = pickle.load(f)
    f.close()
    
    columns = ['q1_vec_'+ str(x) for x in range(300)]
    columns += ['q2_vec_'+str(x) for x in range(300)]
    
    features = pd.DataFrame(columns = columns, index = df.index)
    
    features.loc[:,:] = [weighted_docvec(x, y, embedding_dict, idfs) for x,y in zip(df.q1_wo_stopwords.tolist(), df.q2_wo_stopwords.tolist())]
    
    return features



def get_feature_importance(clf = None, clf_name = None):
    if clf:
        return clf.booster().get_fscore()
    elif clf_name:
        if '.pickle' in clf_name:
            pass
        else:
            clf_name += '.pickle'
        with open('clfs/'+clf_name, 'rb') as f:
            clf = pickle.load(f)
        f.close()
        return clf.booster().get_fscore()
    

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

# TODO: create report of wrong predictions. 
def report_wrong_preds(y_true, y_hat, y_hat_prob, df):
    
    prediction_report = pd.DataFrame(index = df.index)
    prediction_report['q1'] = df['q1']
    prediction_report['q2'] = df['q2']
    prediction_report['y_true'] = y_true
    prediction_report['y_hat'] = y_hat
    prediction_report['y_hat_prob_0'] = y_hat_prob[0]
    prediction_report['y_hat_prob_1'] = y_hat_prob[1]
    prediction_report['diff'] = prediction_report['y_true']-prediction_report['y_hat_prob_1']
    prediction_report.sort_values('diff', acsending = False, inplace = True, axis = 0)
    
    report_name = 'prediction_report_{}.pickle'.format(time.strftime('%m-%d-%H'))
    prediction_report.to_pickle('data/{}'.format(report_name))
    
    print('Prediction report pickled to {}'.format(report_name))
    
    
    return


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
    y_all_hat_prob = clf.predict_proba(features)
    y_all_hat = clf.predict(features)
    
    
    logloss_train = log_loss(y_train, y_train_hat)
    logloss_valid = log_loss(y_valid, y_valid_hat)
    
    return clf, y_all_hat, y_all_hat_prob, logloss_train, logloss_valid


def create_prediction_file(features_test, clf, sub_name):
    
    predictions = pd.DataFrame(index = features_test.index)
    predictions.index.name = 'test_id'
    predictions['is_duplicate'] = clf.predict_proba(features_test)[0]
    
    sub_name = sub_name
    predictions.to_pickle('submissions/{}.pickle'.format(sub_name))
    
    return

def main():
    # Mode defining parameters
    testing = False
    sample = False
    if not testing:

        assert not sample
        
    # Use preexisting features files, if specified
    train_features_file = None
    test_features_file = None
    #train_features_file = 'data/features.pickle'
    #test_features_file = 'data/features_test.pickle'
    
    if testing:
        if sample:
            print('Mode: Testing on Sample')
        else:
            print('Mode: Testing')
    else:
        print('Mode: Predicting')
    print('Generating features started @ {}'.format(time.strftime('%H:%M', time.localtime())))
    
    # Load data based on mode
    t_0 = time.time()
    if not testing:
        train_df, test_df = load_data_from_pickle(test = testing)
        t_1 = time.time()
        print('Train & Test Loaded in {:.2f}s'.format(t_1-t_0))
    else:
        train_df = load_data(test = not testing)
        t_1 = time.time()
        print('Train Loaded in {:.2f}s'.format(t_1-t_0))
        
    # Load embedding
    t_0 = time.time()
    embedding_name = 'wiki.en.vec.pickle'
    with open('embeddings/{}'.format(embedding_name), 'rb') as f:
        embedding_dict = pickle.load(f)
    f.close()
    t_1 = time.time()
    print('Embedding loaded from {} in {:.2f}s @ {}'.format(embedding_name, t_1-t_0, time.strftime('%H:%M', time.localtime())))
        
    # Generate features if features file not provided
    if train_features_file == None:
        if sample:
            train_df = train_df.sample(n = 25000)
        print('Generating Train Features...')
        features = feature_gen(train_df, embedding_dict)
        assert features.shape[0] == train_df.shape[0]
        print('features with {} rows'.format(features.shape[0]))
        features_docvec = docvec_feature_gen(train_df, embedding_dict)
        print('features_docvec with {} rows'.format(features_docvec.shape[0]))
        features = pd.concat([features, features_docvec])
        print('features generated with shape {}'.format(features.shape))
        # Save features to csv
        if not sample:
            features.to_pickle('data/features.pickle')
            print('Train features saved to features.pickle')
        if sample:
            features.to_pickle('data/features_sample.pickle')
            print('Sample of Train features saved to features_sample.csv')
    else:
        features = pd.read_pickle(train_features_file)
        print('features_train loaded from {}'.format(train_features_file))
    
    # Generate/load test features if in prediction mode
    if not testing:
        if test_features_file == None: 
            print('Generating Test Features...')
            features_test = feature_gen(test_df, embedding_dict)
            assert features_test.shape[0] == test_df.shape[0]
            print('test features generated with {} rows'.format(features_test.shape[0]))
            features_docvec_test = docvec_feature_gen(test_df, embedding_dict)
            print('docvec test features generated with {} rows'.format(features_docvec_test.shape[0]))
            assert features_docvec_test.shape[0] == test_df.shape[0]
            features = pd.concat([features_test, features_docvec_test])
            features_test.to_pickle('data/features_test.pickle')
            print('Test features saved to features_test.pickle')
        else:
            features_test = pd.read_pickle(test_features_file)
            print('features_test loaded from {}'.format(test_features_file))
    
        
    # Define labels
    labels = train_df['is_duplicate']
    
    assert features.shape[0] == labels.shape[0]
    
    # Train and Predict!
    clf, y_all_hat, y_all_hat_prob, logloss_train, logloss_valid = xgb_clf(features, labels)
    
    print('Training Logloss: {}'.format(logloss_train))
    print("Validation Logloss: {}".format(logloss_valid))
    
    # Pickle classifier
    if not sample:
        clf_name = 'xgb_{0}_{1:.3f}'.format(time.strftime('%m-%d-%H-%M'), logloss_valid)+'.pickle'
        dump_classifier(clf, clf_name)
    
    # Get and print feature importance
    feature_importance = get_feature_importance(clf = clf)
    print(feature_importance)
    
    # Create prediction file
    
    if not testing:
        sub_name = 'predictions_{:.3f}_{}'.format(logloss_valid, time.strftime('%m-%d-%H-%M'))
        create_prediction_file(features_test, clf, sub_name)
    '''
    # Create and save report highlighting wrong predictions
    report_wrong_preds(labels, y_all_hat, y_all_hat_prob, train_df)
    '''
    
    
    



if __name__ == '__main__':    
    main()
    