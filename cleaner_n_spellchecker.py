#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 05:09:02 2017

@author: ucalegon

Spell Check Based on code by Peter Norvig
http://norvig.com/spell-correct.html

Text Cleaning based on Kernels found on kaggle, e.g.:
https://www.kaggle.com/lystdo/lb-0-18-lstm-with-glove-and-magic-features/code


"""
import re, time, string, gc
import pandas as pd
from gensim import models
from tqdm import tqdm
from collections import Counter
from nltk.corpus import reuters, gutenberg, brown, webtext, inaugural, names, words

tqdm.pandas('tqdm_gui')

# Reuters + Gutenberg contains 62,162 unique words
# Reuters + Gutenberg + Brown + Webtext + Inaugural contains 93,188 words
t_0 = time.time()
#embedding_name = 'glove.42B.300d.w2v.txt'
#embedding_name = 'glove.6B.50d.w2v.txt'
#embedding_name = 'GoogleNews-vectors-negative300.bin'
embedding_name = 'wiki.en.vec'
if '.bin' in embedding_name:
    binary = True
else:
    binary = False
print('Loading word embedding: {}'.format(embedding_name))
word_vec = models.KeyedVectors.load_word2vec_format('embeddings/{}'.format(embedding_name), binary = binary)
WORDS = reuters.words()+gutenberg.words()+brown.words()+webtext.words()+inaugural.words()+names.words()+words.words()+list(word_vec.vocab.keys())
WORDS = Counter(WORDS)
WORDS_set = set(WORDS.keys())
gc.collect()
print('Word universe loaded from corpora in {:.2f}'.format(time.time()-t_0))


def words(text): return re.findall(r'\w+', text.lower())


def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def spell_check(str1, change_log):
    str1_final = str1
    
    str1_final = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", str1_final)
    str1_final = re.sub(r"what's", "what is ", str1_final)
    str1_final = re.sub(r"\'s", " ", str1_final)
    str1_final = re.sub(r"\'ve", " have ", str1_final)
    str1_final = re.sub(r"can't", "cannot ", str1_final)
    str1_final = re.sub(r"n't", " not ", str1_final)
    str1_final = re.sub(r"i'm", "i am ", str1_final)
    str1_final = re.sub(r"\'re", " are ", str1_final)
    str1_final = re.sub(r"\'d", " would ", str1_final)
    str1_final = re.sub(r"\'ll", " will ", str1_final)
    str1_final = re.sub(r"(\d+)(k)", r"\g<1>000", str1_final)
    str1_final = re.sub(r" e g ", " eg ", str1_final)
    str1_final = re.sub(r"e - mail", "email", str1_final)
    
    current_check = {}
    
    # Check for acronyms, punctuation, math/algebra.
    str1_split = str1_final.split(' ')
    for w_og in str1_split:
        w = w_og.strip(string.punctuation).strip(string.punctuation)
        if (not w.isupper()) & (w.isalpha()):
            if (w not in WORDS.keys()) & (w.lower() not in WORDS.keys()):
                if w in current_check.keys():
                    w_corrected = current_check.get(w)
                else:
                    w_corrected = correction(w)
                    if (w_corrected != w_og) & (w_corrected in word_vec.vocab):
                        str1_final.replace(w, w_corrected)
                        current_check[w] = w_corrected
                        #print(w, w_corrected)
                        change_log.write(str(w+'    '+w_corrected))
                        change_log.write('\n')
            
    
    return str1_final
    
    


def main():
    t_1 = time.time()
    train_df = pd.read_csv('data/train.csv', index_col = 'id')
    train_df.fillna(value = '', inplace = True)
    t_2 = time.time()
    print('Train loaded in {:.2f}'.format(t_2 - t_1))
    
    # TODO: Add tqdm integration for df.apply => df.progress_apply
    
    change_log = open('data/spell_check_log_{}.txt'.format(time.strftime('%m-%d-%H')), 'w')
    change_log.write('TRAIN SPELL CHECK CHANGES')
    change_log.write('\n')
    train_df['question1'] = train_df.progress_apply(lambda x: spell_check(x['question1'], change_log), axis = 1)
    train_df['question2'] = train_df.progress_apply(lambda x: spell_check(x['question2'], change_log), axis = 1)
    train_df.to_csv('data/train_checked.csv')
    t_4 = time.time()
    print('Spell check run on train completed in {:.2f}s'.format(t_4-t_2))
    '''
    change_log.write('\n')
    change_log.write('TEST SPELL CHECK CHANGES')
    change_log.write('\n')
    test_df = pd.read_csv('data/test.csv', index_col = 'test_id')
    test_df.fillna(value = '', inplace = True)
    t_5 = time.time()
    print('Test loaded in {:.2f}'.format(t_5-t_4))
    test_df['question1'] = test_df.progress_apply(lambda x: spell_check(x['question1'], change_log), axis = 1)
    test_df['question2'] = test_df.progress_apply(lambda x: spell_check(x['question2'], change_log), axis = 1)
    test_df.to_csv('data/test_checked.csv')
    t_6 = time.time()
    print('Spell check run on test completed in {:.2f}s'.format(t_6-t_5))
    '''
    change_log.close()
    
    
if __name__ == '__main__':
    
    main()
    
    