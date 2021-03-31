import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.model_selection import cross_val_score

clipData = pd.read_csv('../data/clipTrainPure.csv', index_col=0)

clipData.dropna(inplace=True)
print(len(clipData))

trainSet = clipData[['nx', 'ny', 'rsx', 'rsy', 'lsx', 'lsy', 'rhx', 'rhy', 'lhx', 'lhy', 'rkx', 'rky', 'lkx', 'lky']]


labels = clipData['label']

svc = svm.SVC(kernel='rbf')

scores = cross_val_score(svc, trainSet, labels, cv=10)
print(scores)
