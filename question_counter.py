#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 23:53:50 2017

implementation motivated by kaggle kernel by SRK @
https://www.kaggle.com/sudalairajkumar/simple-leaky-exploration-notebook-quora


@author: ucalegon
"""
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

train_df = pd.read_csv('data/train_checked.csv', index_col = 'id')
train_df.fillna(value = '', inplace = True)
test_df = pd.read_csv('data/test_checked.csv', index_col = 'test_id')
train_df.fillna(value = '', inplace = True)

question_df = pd.concat([train_df[['question1', 'question2']], test_df[['question1', 'question2']]], axis = 0).reset_index(drop= 'index')

q_dict = defaultdict(set)
for i in tqdm(range(question_df.shape[0])):
    q_dict[question_df.at[i, 'question1']].add(question_df.at[i, 'question2'])
    q_dict[question_df.at[i, 'question2']].add(question_df.at[i, 'question1'])
    
del question_df
    
    
def q1_freq(row):
    return (len(q_dict[row['question1']]))
    
def q2_freq(row):
    return (len(q_dict[row['question2']]))
    
def q1_q2_intersect(row):
    return (len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))


tqdm.pandas(desc = 'train q1-q2')
train_df['q1_q2_intersect'] = train_df.progress_apply(lambda x: q1_q2_intersect(x), axis=1)
tqdm.pandas(desc = 'train q1')
train_df['q1_freq'] = train_df.progress_apply(lambda x: q1_freq(x), axis=1)
tqdm.pandas(desc = 'train q2')
train_df['q2_freq'] = train_df.progress_apply(lambda x: q2_freq(x), axis=1)

tqdm.pandas(desc = 'test q1-q2')
test_df['q1_q2_intersect'] = test_df.progress_apply(lambda x: q1_q2_intersect(x), axis=1)
tqdm.pandas(desc = 'test q1')
test_df['q1_freq'] = test_df.progress_apply(lambda x: q1_freq(x), axis=1)
tqdm.pandas(desc = 'test q2')
test_df['q2_freq'] = test_df.progress_apply(lambda x: q2_freq(x), axis=1)
    
# Deprecated
'''
question_list = pd.concat([train_df['question1'], train_df['question2'], test_df['question1'], test_df['question2']]).to_list()


q_counter = collections.Counter(question_list)

q_count_df = pd.DataFrame.from_dict(q_counter, orient = 'index').reset_index()

q_count_df.rename(columns = {'index':'question1', 0:'q1_count'}, inplace = True)

train_df = train_df.merge(q_count_df, how = 'inner', on = 'question1')
test_df = test_df.merge(q_count_df, how = 'inner', on = 'question1')

q_count_df.rename(columns = {'question1':'question2', 'q1_count':'q2_count'}, inplace = True)

train_df = train_df.merge(q_count_df, how = 'inner', on = 'question2')
test_df = test_df.merge(q_count_df, how = 'inner', on = 'question2')

with open('data/q_counter.pickle', 'wb') as f:
    pickle.dump(q_counter, f)
f.close()
'''

train_df.index.name = 'id'
test_df.index.name = 'test_id'
train_df[['q1_freq', 'q2_freq', 'q1_q2_intersect']].to_csv('data/train_checked_q_count.csv')
test_df[['q1_freq', 'q2_freq', 'q1_q2_intersect']].to_csv('data/test_checked_q_count.csv')












