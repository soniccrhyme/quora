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
            train_df[key] = train_df.apply(lambda x: ast.literal_eval(x[key]), axis = 1)
            test_df[key] = test_df.apply(lambda x: ast.literal_eval(x[key]), axis = 1 )
        return train_df, test_df
    else:
        for key in ext_keys:
            train_df[key] = train_df.apply(lambda x: ast.literal_eval(x[key]), axis = 1)
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
    
    
    
    return
        

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


def feature_gen(df):
    '''
    Generate features given a dataframe of questions, tokens, etc.
    '''
    features = pd.DataFrame(index = df.index)
    
    t_0 = time.time()
        
    # Jaccard features
    features['jaccard_full'] = df.apply(lambda x: jaccard(x['q1_token'], x['q2_token']), axis = 1)
    features['jaccard_wo_stop'] = df.apply(lambda x: jaccard(x['q1_wo_stopwords'], x['q2_wo_stopwords']), axis = 1) 
    features['jaccard_of_stop'] = df.apply(lambda x: jaccard(x['q1_stopwords'], x['q2_stopwords']), axis = 1)
    features['jaccard_specchar'] = df.apply(lambda x: jaccard(x['q1_specchar'], x['q2_specchar']), axis = 1)
    t_1 = time.time()
    print('Jaccard-based features generated in {:.2f}s'.format(t_1-t_0))
    print(features.shape)
    assert features.shape[0] == df.shape[0]
    
    # Fuzzywuzzy features
    features['fuzz_QRatio_full'] = df.apply(lambda x: fuzz.QRatio(x['q1'], x['q2'])/100., axis = 1)
    features['fuzz_QRatio_tks'] = df.apply(lambda x: fuzz.QRatio(' '.join(x['q1_token']), ' '.join(x['q2_token']))/100., axis = 1)
    features['fuzz_WRatio_full'] = df.apply(lambda x: fuzz.WRatio(x['q1'], x['q2'])/100., axis = 1)
    features['fuzz_WRatio_tks'] = df.apply(lambda x: fuzz.WRatio(' '.join(x['q1_token']), ' '.join(x['q2_token']))/100., axis = 1)
    features['fuzz_partial_full'] = df.apply(lambda x: fuzz.partial_ratio(x['q1'], x['q2'])/100., axis = 1)
    features['fuzz_partial_tks'] = df.apply(lambda x: fuzz.partial_ratio(' '.join(x['q1_token']), ' '.join(x['q2_token']))/100., axis = 1)
    features['fuzz_partknset_full'] = df.apply(lambda x: fuzz.partial_token_set_ratio(x['q1'], x['q2'])/100., axis = 1)
    features['fuzz_partknset_tks'] = df.apply(lambda x: fuzz.partial_token_set_ratio(' '.join(x['q1_token']), ' '.join(x['q2_token']))/100., axis = 1)
    features['fuzz_partknsort_full'] = df.apply(lambda x: fuzz.partial_token_sort_ratio(x['q1'], x['q2'])/100., axis = 1)
    features['fuzz_partknsort_tks'] = df.apply(lambda x: fuzz.partial_token_sort_ratio(' '.join(x['q1_token']), ' '.join(x['q2_token']))/100., axis = 1)
    features['fuzz_tknset_full'] = df.apply(lambda x: fuzz.token_set_ratio(x['q1'], x['q2'])/100., axis = 1)
    features['fuzz_tknset_tks'] = df.apply(lambda x: fuzz.token_set_ratio(' '.join(x['q1_token']), ' '.join(x['q2_token']))/100., axis = 1)
    features['fuzz_tknsort_full'] = df.apply(lambda x: fuzz.token_sort_ratio(x['q1'], x['q2'])/100., axis = 1)
    features['fuzz_tknsort_tks'] = df.apply(lambda x: fuzz.token_sort_ratio(' '.join(x['q1_token']), ' '.join(x['q2_token']))/100., axis = 1)
    t_1_2 = time.time()
    print('Fuzzywuzzy based features generated in {:.2f}'.format(t_1_2-t_1))
    print(features.shape)
    assert features.shape[0] == df.shape[0]
    
    
    nlp = spacy.load('en')
    embedding_name = 'wiki.en.vec.pickle'
    with open('embeddings/{}'.format(embedding_name), 'rb') as f:
        embedding_dict = pickle.load(f)
    f.close()
    t_1_3 = time.time()
    print('Embedding loaded from {} in {:.2f}s'.format(embedding_name, t_1_3-t_1_2))
    # Embedding similarity features, via Spacy
    features['similarity_spacy_full'] = df.apply(lambda x: nlp(' '.join(x['q1_token'])).similarity(nlp(' '.join(x['q2_token']))), axis = 1)
    features['similarity_spacy_unique'] = df.apply(lambda x: nlp(' '.join(sep_unique(x['q1_token'], x['q2_token']))).similarity(nlp(' '.join(sep_unique(x['q2_token'], x['q1_token'])))), axis = 1)
    features['sim_min_cosine'] = df.apply(lambda x: vec_similarity(x['q1_token'], x['q2_token'], embedding_dict, agg_type = 'min'), axis = 1)
    features['sim_max_cosine'] = df.apply(lambda x: vec_similarity(x['q1_token'], x['q2_token'], embedding_dict, agg_type = 'max'), axis = 1)
    features['sim_avg_euclid'] = df.apply(lambda x: vec_similarity(x['q1_token'], x['q2_token'], embedding_dict, agg_type = 'avg', sim_type = 'euclidean'), axis = 1)
    features['sim_max_euclid'] = df.apply(lambda x: vec_similarity(x['q1_token'], x['q2_token'], embedding_dict, agg_type = 'max', sim_type = 'euclidean'), axis = 1)
    features['sim_min_euclid'] = df.apply(lambda x: vec_similarity(x['q1_token'], x['q2_token'], embedding_dict, agg_type = 'min', sim_type = 'euclidean'), axis = 1)
    features['sim_avg_manhattan'] = df.apply(lambda x: vec_similarity(x['q1_token'], x['q2_token'], embedding_dict, agg_type = 'avg', sim_type = 'manhattan'), axis = 1)
    features['sim_max_manhattan'] = df.apply(lambda x: vec_similarity(x['q1_token'], x['q2_token'], embedding_dict, agg_type = 'max', sim_type = 'manhattan'), axis = 1)
    features['sim_min_manhattan'] = df.apply(lambda x: vec_similarity(x['q1_token'], x['q2_token'], embedding_dict, agg_type = 'min', sim_type = 'manhattan'), axis = 1)
    features['sim_avg_correlation'] = df.apply(lambda x: vec_similarity(x['q1_token'], x['q2_token'], embedding_dict, agg_type = 'avg', sim_type = 'correlation'), axis = 1)
    features['sim_max_correlation'] = df.apply(lambda x: vec_similarity(x['q1_token'], x['q2_token'], embedding_dict, agg_type = 'max', sim_type = 'correlation'), axis = 1)
    features['sim_min_correlation'] = df.apply(lambda x: vec_similarity(x['q1_token'], x['q2_token'], embedding_dict, agg_type = 'min', sim_type = 'correlation'), axis = 1)
    print(features.shape)
    assert features.shape[0] == df.shape[0]
    # Load idfs for weighted_averages
    with open('data/idf_weights.pickle', 'rb') as f:
        idfs = pickle.load(f)
    f.close()
    features['sim_weighted_avg_cosine'] = df.apply(lambda x: weighted_vec_similarity(x['q1_token'], x['q2_token'], embedding_dict, idfs, sim_type = 'cosine'), axis = 1)
    features['sim_weighted_avg_euclid'] = df.apply(lambda x: weighted_vec_similarity(x['q1_token'], x['q2_token'], embedding_dict, idfs, sim_type = 'euclidean'), axis = 1)
    features['sim_weighted_avg_manhattan'] = df.apply(lambda x: weighted_vec_similarity(x['q1_token'], x['q2_token'], embedding_dict, idfs, sim_type = 'manhattan'), axis = 1)
    features['sim_weighted_avg_correlation'] = df.apply(lambda x: weighted_vec_similarity(x['q1_token'], x['q2_token'], embedding_dict, idfs, sim_type = 'correlation'), axis = 1)
    t_2 = time.time()
    print('Spacy-based similarity features generated in {:.2f}s'.format(t_2-t_1_2))
    print(features.shape)
    assert features.shape[0] == df.shape[0]
    
    
    
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
    print('Question frequency features added in {:.2f}s'.format(t_5-t_2))
    print(features.shape)
    assert features.shape[0] == df.shape[0]
    
    
    # Word-Share Features
    features['q1_in_q2'] = df.apply(lambda x: word_share(x['q1_token'], x['q2_token']), axis = 1)
    features['q1_in_q2'] = df.apply(lambda x: word_share(x['q2_token'], x['q1_token']), axis = 1)
    
    # Generic count features
    features['q1_word_count'] = df.apply(lambda x: len(x['q1_token']), axis = 1)
    features['q2_word_count'] = df.apply(lambda x: len(x['q2_token']), axis = 1)
    features['q1_char_len'] = df.apply(lambda x: len(x['q1'].replace(' ', '')), axis = 1)
    features['q2_char_len'] = df.apply(lambda x: len(x['q2'].replace(' ', '')), axis = 1)
    features['q1_len'] = df.apply(lambda x: len(x['q1']), axis = 1)
    features['q2_len'] = df.apply(lambda x: len(x['q2']), axis = 1)
    
    t_end = time.time()
    print('All features generated in {:.2f}; {} rows'.format(t_end-t_0, features.shape[0]))
    print(features.shape)
    assert features.shape[0] == df.shape[0]
    
    
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


def keras_clf(features, labels, clf_pickle = None):
    
    
    
    
    
    
    
    
    
    return 



def create_prediction_file(features_test, clf, sub_name):
    
    predictions = pd.DataFrame(index = features_test.index)
    predictions.index.name = 'test_id'
    predictions['is_duplicate'] = clf.predict_proba(features_test)[0]
    
    sub_name = sub_name
    predictions.to_csv('submissions/{}.csv'.format(sub_name), )
    
    return

def main():
    # Mode defining parameters
    testing = True
    sample = True
    if not testing:
        assert not sample
        
    # Use preexisting features files, if specified
    train_features_file = None
    test_features_file = None
    #train_features_file = 'data/features.csv'
    #test_features_file = 'data/features_test.csv'
    
    if testing:
        if sample:
            print('Mode: Testing on Sample')
        else:
            print('Mode: Testing')
    else:
        print('Mode: Predicting')
    
    # Load data based on mode
    t_0 = time.time()
    if not testing:
        train_df, test_df = load_data(test = not testing)
        t_1 = time.time()
        print('Train & Test Loaded in {:.2f}s'.format(t_1-t_0))
    else:
        train_df = load_data(test = not testing)
        t_1 = time.time()
        print('Train Loaded in {:.2f}s'.format(t_1-t_0))
        
    # Generate features if features file not provided
    if train_features_file == None:
        if sample:
            train_df = train_df.sample(n = 25000)
        features = feature_gen(train_df)
        assert features.shape[0] == train_df.shape[0]
        # Save features to csv
        if not sample:
            features.to_csv('data/features.csv')
            print('Train features saved to features.csv')
        if sample:
            features.to_csv('data/features_sample.csv')
            print('Sample of Train features saved to features_sample.csv')
    else:
        features = pd.read_csv(train_features_file, index_col = 'id')
        print('features_train loaded from {}'.format(train_features_file))
    
    if not testing:
        if test_features_file == None: 
            features_test = feature_gen(test_df)
            features_test.to_csv('data/features_test.csv')
            print('Test features saved to features_test.csv')
        else:
            features_test = pd.read_csv(test_features_file, index_col = 'id')
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
    
    # Create and save report highlighting wrong predictions
    report_wrong_preds(labels, y_all_hat, y_all_hat_prob, train_df)
    
    
    
    



if __name__ == '__main__':    
    main()
    