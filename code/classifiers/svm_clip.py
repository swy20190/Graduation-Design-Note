import pandas as pd
import numpy as np
from math import isnan

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve

from evaluator import confusion_metric_drawer
from matplotlib import pyplot as plt

clipData = pd.read_csv('../data/clipTrainPure.csv', index_col=0)


trainSet = clipData[['nx', 'ny', 'rsx', 'rsy', 'lsx', 'lsy', 'rhx', 'rhy', 'lhx', 'lhy', 'rkx', 'rky', 'lkx', 'lky']]


labelTrain = clipData['label']
# print(np.array(labelTrain).tolist())


svc = svm.SVC(kernel='rbf')
svc.fit(trainSet, labelTrain)

testDf = pd.read_csv('../data/clipTestPure.csv', index_col=0)
testSet = testDf[['nx', 'ny', 'rsx', 'rsy', 'lsx', 'lsy', 'rhx', 'rhy', 'lhx', 'lhy', 'rkx', 'rky', 'lkx', 'lky']]
labelPre = svc.predict(testSet)
labelInd = testDf['original_index']
labelTru = testDf['label']

preIndexList = []
for i in range(len(labelPre)):
    curr = [labelPre[i], labelInd[i]]
    preIndexList.append(curr)

labelDf = pd.DataFrame(preIndexList, columns=['label', 'index'])
labelDf.to_csv('../data/labelTru.csv')

metric = confusion_matrix(labelTru, labelPre)

print(metric)

confusion_metric_drawer(metric, "../output/metric_clip.png")

# draw ROC curve
testSet = np.array(testSet).tolist()
scores = svc.decision_function(testSet)

fpr, tpr, thresholds = roc_curve(labelTru, scores)

plt.plot(fpr, tpr, color='cyan', label='clip_difference')

plt.legend()
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.title('ROC curves for fall detection classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.savefig("../output/roc_curve_clip.png")
plt.show()
plt.cla()
