from __future__ import division
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale

sys.path.append('models/word2vec_twitter_model')
from word2vecReader import Word2Vec

model_path = 'models/word2vec_twitter_model/word2vec_twitter_model.bin'

print 'Loading the model...'
model = Word2Vec.load_word2vec_format(model_path, binary=True)

def preprocess(text):
    special_chars = """.,?!:;(){}[]#"""
    for c in special_chars:
        text = text.replace(c, ' %s '%c)
    words = text.lower().split()

    return words

def get_vector(text, model=model, size=400):
    words = preprocess(text)
    vec = np.zeros(size)
    count = 0.
    for word in words:
        try:
            vec += model[word]
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

X_pos = open('data/positive-all','r').readlines()
X_neu = open('data/neutral-all','r').readlines()
X_neg = open('data/negative-all','r').readlines()


X_pos_vec = np.array(map(get_vector, X_pos))
X_neu_vec = np.array(map(get_vector, X_neu))
X_neg_vec = np.array(map(get_vector, X_neg))


y_pos_vec = np.ones(len(X_pos_vec))
y_neu_vec = np.zeros(len(X_neu_vec))
y_neg_vec = np.full(len(X_neg_vec),-1)

X_all = np.concatenate((X_pos_vec, X_neu_vec, X_neg_vec))
y_all = np.concatenate((y_pos_vec, y_neu_vec, y_neg_vec))

X_all = scale(X_all)

X_all.dump('models/X_all_w2v')
y_all.dump('models/y_all_w2v')


X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.4, random_state=42)

logit = LogisticRegression(C=0.5)
clf = logit.fit(X_train, y_train)
pred = clf.predict(X_test)
print classification_report(y_test, pred, target_names=['1.','0.','-1.'])
