#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 21:12:25 2017

Calculate idf-weights

@author: ucalegon
"""

import pickle, time
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
from collections import Counter

def dump_idf(idf_weights):
    with open('data/idf_weights.pickle', 'wb') as f:
        pickle.dump(idf_weights, f)
    f.close()
    print('idf weights pickled to data/idf_weights.pickle')
    return
    

def cheap_idf(question_list):
    # Create question list of word sets & corpora word set
    t_1 = time.time()
    tokenizer = RegexpTokenizer(r'\w+')
    tqdm.pandas(desc = 'apply_bar')
    corpus_size = len(question_list)
    word_count = Counter(tokenizer.tokenize(' '.join(question_list)))
    t_2 = time.time()
    print('Word Count dict generated in {}s'.format(t_2-t_1))
    
    
    idf_weights = {}
    eps = 1
    t_3 = time.time()
    print('Calculating cheap idfs...')
    for w in tqdm(word_count.keys()):
        idf_weights[w] = eps + np.log10((corpus_size+eps)/(word_count[w]+eps))
    t_4 = time.time()
    print('cheap idf weights done calculating in {:.2f}'.format(t_4-t_3))
    
    dump_idf(idf_weights)
    
        
    


def expensive_idf(question_list):
    
    # Create question list of word sets & corpora word set
    t_1 = time.time()
    tokenizer = RegexpTokenizer(r'\w+')
    tqdm.pandas(desc = 'apply_bar')
    corpus_size = len(question_list)
    word_set = set(tokenizer.tokenize(' '.join(question_list)))
    t_2 = time.time()
    print('Word Set generated in {}s'.format(t_2-t_1))
    
    
    # Use epsilon as smoothing factor
    epsilon = 1
    
    idf_weights = {}
    
    t_3 = time.time()
    print('Calculating idf for all words...')
    
    def idf_gen(word_set, question_list):
        for w in tqdm(word_set):
            freq = 0
            for q in question_list:
                if w in q:
                    freq += 1
            idf = epsilon + np.log10((corpus_size + epsilon)/(freq+epsilon))
            idf_weights[w] = idf
            dump_idf(idf_weights)
    
    
    import threading
    threading.Thread(target = idf_gen, args = (word_set, question_list)).start()
    
    t_4 = time.time()
    print('idf wieghts calculated in {:.2f}'.format(t_4-t_3))
    
    return
    


def main():
    # 'hard', accuracte idf or 'easy', fast idf
    mode = 'easy'
    
    # Import data
    t_0 = time.time()
    train_df = pd.read_csv('data/train_checked.csv', index_col = 'id' )
    train_df.rename(columns = {'question1':'q1', 'question2':'q2'}, inplace = True)
    train_df.fillna(value = '', inplace = True)
    test_df = pd.read_csv('data/test_checked.csv', index_col = 'test_id')
    test_df.rename(columns = {'question1':'q1', 'question2':'q2'}, inplace = True)
    test_df.fillna(value = '', inplace = True)
    t_1 = time.time()
    print('Train and Test loaded in {:.2f}s'.format(t_1-t_0))
    
    # Create question list of word sets & corpora word set
    
    question_list = pd.concat([train_df.q1, train_df.q2, test_df.q1, test_df.q2]).tolist()
    
    if mode == 'hard':
        expensive_idf(question_list)
    else:
        cheap_idf(question_list)
        
if __name__ == '__main__':
    main()


