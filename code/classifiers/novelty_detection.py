# This file classify data using novelty and outlier detection
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


def confusion_metric_drawer_novelty(metric_array, plt_name):
    TP = metric_array[0][0]
    FP = metric_array[1][0]
    FN = metric_array[0][1]
    TN = metric_array[1][1]

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    fig, ax = plt.subplots(figsize=(3.5, 6))
    ax.matshow(metric_array, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(metric_array.shape[0]):
        for j in range(metric_array.shape[1]):
            ax.text(x=j, y=i, s=metric_array[i, j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    font = {'color': 'red',
            'size': 10,
            'family': 'Times New Roman', }
    plt.text(-0.5, -1.5, 'Precision: '+str(precision), fontdict=font)
    plt.text(-0.5, -1, 'Recall: '+str(recall), fontdict=font)

    plt.savefig(plt_name)
    plt.show()
    plt.cla()


# generate training set
# note: all the training set must be ADL
ADL_df = pd.read_csv("../data/normalized/ADLNormThigh.csv")
# select features
training_df = ADL_df[['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '5x', '5y', '6x', '6y', '8x', '8y', '9x', '9y',
                      '11x', '11y', '12x', '12y']]
training_data = np.array(training_df).tolist()

# fit the model
clf = svm.OneClassSVM(kernel='rbf', nu=0.01)
clf.fit(training_data)

# generate the test set
ADLTest_df = pd.read_csv("../data/normalized/ADLNormThighTest.csv")
ADLTest_df = ADLTest_df[['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '5x', '5y', '6x', '6y', '8x', '8y', '9x', '9y',
                         '11x', '11y', '12x', '12y']]
ADLTest_data = np.array(ADLTest_df).tolist()

FallTest_df = pd.read_csv("../data/normalized/FallNormThighTest.csv")
FallTest_df = FallTest_df[['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '5x', '5y', '6x', '6y', '8x', '8y', '9x', '9y',
                           '11x', '11y', '12x', '12y']]
FallTest_data = np.array(FallTest_df).tolist()

num_norm = len(ADLTest_data)
num_novel = len(FallTest_data)

test_label = []
for i in range(num_norm):
    test_label.append(1)
for i in range(num_novel):
    test_label.append(-1)

test_data = ADLTest_data + FallTest_data

predict_label = clf.predict(test_data)
metric = confusion_matrix(test_label, predict_label)

confusion_metric_drawer_novelty(metric, "../output/metric_thigh_OCSVM.png")
