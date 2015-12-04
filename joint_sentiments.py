from __future__ import division
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale


X_w2v = np.load('models/X_all_w2v')
X_lsa = np.load('models/X_all_lsa')
y = np.load('models/y_all_w2v')

X = np.hstack((X_w2v,X_lsa))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

logit = LogisticRegression(C=0.5)
clf = logit.fit(X_train, y_train)
pred = clf.predict(X_test)
print classification_report(y_test, pred, target_names=['1.','0.','-1.'])
