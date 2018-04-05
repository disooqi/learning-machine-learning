import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


data = np.loadtxt('../data/ex2data1.txt', delimiter=',')
features = data[:,:2]
y = data[:,2]

logreg = LogisticRegression(C=1e5)
y = y.ravel()
logreg.fit(features, y).score(features, y)

p = logreg.predict(features)
B = np.zeros_like(y)
B[p==y]=1
Acc = 100* B.sum()/B.size
print(Acc)

print(OneVsRestClassifier(LogisticRegression(penalty='l1')).fit(features, y).score(features, y))
print(OneVsRestClassifier(LogisticRegression(penalty='l2')).fit(features, y).score(features, y))
print(OneVsRestClassifier(LogisticRegression(C=1e5)).fit(features, y).score(features, y))
print(OneVsRestClassifier(LogisticRegression(C=10)).fit(features, y).score(features, y))