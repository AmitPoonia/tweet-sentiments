from __future__ import division
import sys
import os
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
from gensim import corpora, models


def wordlist(text, remove_stopwords=True):
    special_chars = """.,?!:;(){}[]"""
    for c in special_chars:
        text = text.replace(c, ' %s '%c)

    words = text.lower().split()
    return words

def preprocess(texts):
    list_of_lists = map(wordlist, texts)
    return list_of_lists

def remove_tuples(tuples_list):
    tupleless = [tup[1] for tup in tuples_list]
    return tupleless


num_topics = 100

X_pos = open("data/positive-all","r").readlines()
X_neu = open("data/neutral-all","r").readlines()
X_neg = open("data/negative-all","r").readlines()

y_pos_vec = np.ones(len(X_pos))
y_neu_vec = np.zeros(len(X_neu))
y_neg_vec = np.full(len(X_neg),-1)


y_all = np.concatenate((y_pos_vec, y_neu_vec, y_neg_vec))

X_processed = preprocess(X_pos+X_neu+X_neg)
dictionary = corpora.Dictionary(X_processed)
corpus = map(dictionary.doc2bow, X_processed)
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)

X_all = np.zeros((len(X_processed), num_topics))
X_all_ = map(remove_tuples, lsi[corpus])

for i,row in enumerate(X_all_):
    for j,col in enumerate(X_all_[i]):
        X_all[i][j] = X_all_[i][j]


X_all = scale(X_all)

X_all.dump('models/X_all_lsa')
y_all.dump('models/y_all_lsa')
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.4, random_state=42)

logit = LogisticRegression(C=0.5)
clf = logit.fit(X_train, y_train)
pred = clf.predict(X_test)
print classification_report(y_test, pred, target_names=['1.','0.','-1.'])
