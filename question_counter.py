#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 23:53:50 2017

@author: ucalegon
"""
import pickle
import pandas as pd
import collections

train_df = pd.read_csv('data/train.csv', index_col = 'id')
train_df.fillna(value = '', inplace = True)
test_df = pd.read_csv('data/test.csv', index_col = 'test_id')
train_df.fillna(value = '', inplace = True)

question_list = []
question_list += train_df['question1'].tolist()
question_list += train_df['question2'].tolist()
question_list += test_df['question1'].tolist()
question_list += test_df['question2'].tolist()

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

train_df.index.name = 'id'
test_df.index.name = 'test_id'
train_df[['q1_count', 'q2_count']].to_csv('data/train_q_count.csv')
test_df[['q1_count', 'q2_count']].to_csv('data/test_q_count.csv')