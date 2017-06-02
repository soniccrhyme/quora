#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 05:09:02 2017

@author: ucalegon

Based on code by Peter Norvig
Found at http://norvig.com/spell-correct.html
"""
import re, time, string, gc
import pandas as pd
import numpy as np
import spacy
from gensim import models
from collections import Counter
from nltk.corpus import reuters, gutenberg, brown, webtext, inaugural, names, words


# Reuters + Gutenberg contains 62,162 unique words
# Reuters + Gutenberg + Brown + Webtext + Inaugural contains 93,188 words
t_0 = time.time()
#embedding_name = 'glove.42B.300d.w2v.txt'
embedding_name = 'glove.6B.50d.w2v.txt'
word_vec = models.KeyedVectors.load_word2vec_format('embeddings/{}'.format(embedding_name), binary = False)
WORDS = reuters.words()+gutenberg.words()+brown.words()+webtext.words()+inaugural.words()+names.words()+words.words()+list(word_vec.vocab.keys())
WORDS = Counter(WORDS)
WORDS_set = set(WORDS.keys())
del word_vec
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
    
    
    
    # Check for acronyms, punctuation, math/algebra.
    str1_split = str1.split(' ')
    for w_og in str1_split:
        w = w_og.strip(string.punctuation).strip(string.punctuation)
        if (not w.isupper()) & (w.isalpha()):
            if (w not in WORDS.keys()) & (w.lower() not in WORDS.keys()):
                w_corrected = correction(w)
                if w_corrected != w_og:
                    str1_final.replace(w, w_corrected)
                    #print(w, w_corrected)
                    change_log.write(str(w+'    '+w_corrected))
                    change_log.write('\n')
        
    
    return str1_final
    
    


def main():
    t_1 = time.time()
    train_df_og = pd.read_csv('data/train.csv', index_col = 'id')
    train_df_og.fillna(value = '', inplace = True)
    t_2 = time.time()
    print('Train loaded in {:.2f}'.format(t_2 - t_1))
    #test_df_og = pd.read_csv('data/test.csv', index_col = 'test_id')
    #test_df_og.fillna(value = '', inplace = True)
    
    t_3 = time.time()
    change_log = open('data/spell_check_log.txt', 'w')
    train_df_og['question1'] = train_df_og.apply(lambda x: spell_check(x['question1'], change_log), axis = 1)
    change_log.close()
    t_4 = time.time()
    print('Spell check run on train completed in {:.2f}s'.format(t_4-t_3))
    
    
    
main()
    
    