# This file evaluates different models on test dataset
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_curve

from matplotlib import pyplot as plt

# generate training set
ADL_train = pd.read_csv("../data/normalized/ADLNormThigh.csv", index_col=0)
ADL_train = ADL_train[['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '5x', '5y', '6x', '6y', '8x', '8y', '9x', '9y',
                       '11x', '11y', '12x', '12y']]
ADL_train = np.array(ADL_train).tolist()
false_num = len(ADL_train)

Fall_train = pd.read_csv("../data/normalized/FallNormThigh.csv", index_col=0)
Fall_train = Fall_train[['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '5x', '5y', '6x', '6y', '8x', '8y', '9x', '9y',
                         '11x', '11y', '12x', '12y']]
Fall_train = np.array(Fall_train).tolist()
true_num = len(Fall_train)

training_set = ADL_train + Fall_train
training_label = []

for i in range(false_num):
    training_label.append(0)
for i in range(true_num):
    training_label.append(1)

# generate test set
ADL_test = pd.read_csv("../data/normalized/ADLNormThighTest.csv", index_col=0)
ADL_test = ADL_test[['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '5x', '5y', '6x', '6y', '8x', '8y', '9x', '9y',
                    '11x', '11y', '12x', '12y']]
ADL_test = np.array(ADL_test).tolist()
false_num = len(ADL_test)

Fall_test = pd.read_csv("../data/normalized/FallNormThighTest.csv", index_col=0)
Fall_test = Fall_test[['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '5x', '5y', '6x', '6y', '8x', '8y', '9x', '9y',
                       '11x', '11y', '12x', '12y']]
Fall_test = np.array(Fall_test).tolist()
true_num = len(Fall_test)

test_set = ADL_test + Fall_test
test_label = []

for i in range(false_num):
    test_label.append(0)
for i in range(true_num):
    test_label.append(1)

svc_thigh = svm.SVC()
svc_thigh.fit(training_set, training_label)
thigh_score = svc_thigh.decision_function(test_set)

fpr, tpr, thresholds = roc_curve(test_label, thigh_score)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.title('ROC curve for fall detection classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()
