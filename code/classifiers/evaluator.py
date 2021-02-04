# This file evaluates different models on test dataset
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt


def roc_drawer(features=None, normalized_mode="thigh_len"):
    """
    this function generates the plot for roc curve, models are svm based on different normalized data
    :param features: features chosen to train the model
    :param normalized_mode: how the data is normalized, e.g. "thigh_len"
    :return: plot TP rate, FP rate, confusion matrix
    """
    if features is None:
        features = ['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '5x', '5y', '6x', '6y', '8x', '8y', '9x', '9y',
                    '11x', '11y', '12x', '12y']

    # read set
    if normalized_mode == "thigh_len":
        ADL_train = pd.read_csv("../data/normalized/ADLNormThigh.csv", index_col=0)
        Fall_train = pd.read_csv("../data/normalized/FallNormThigh.csv", index_col=0)
        ADL_test = pd.read_csv("../data/normalized/ADLNormThighTest.csv", index_col=0)
        Fall_test = pd.read_csv("../data/normalized/FallNormThighTest.csv", index_col=0)
    elif normalized_mode == "torso_box":
        ADL_train = pd.read_csv("../data/normalized/ADLNormTorso.csv", index_col=0)
        Fall_train = pd.read_csv("../data/normalized/FallNormTorso.csv", index_col=0)
        ADL_test = pd.read_csv("../data/normalized/ADLNormTorsoTest.csv", index_col=0)
        Fall_test = pd.read_csv("../data/normalized/FallNormTorsoTest.csv", index_col=0)
    else:
        ADL_train = pd.read_csv("../data/normalized/ADLNormNone.csv", index_col=0)
        Fall_train = pd.read_csv("../data/normalized/FallNormNone.csv", index_col=0)
        ADL_test = pd.read_csv("../data/normalized/ADLNormNoneTest.csv", index_col=0)
        Fall_test = pd.read_csv("../data/normalized/FallNormNoneTest.csv", index_col=0)

    # generate the training set
    ADL_train = ADL_train[features]
    ADL_train = np.array(ADL_train).tolist()
    false_num = len(ADL_train)

    Fall_train = Fall_train[features]
    Fall_train = np.array(Fall_train).tolist()
    true_num = len(Fall_train)

    training_set = ADL_train + Fall_train
    training_label = []

    for i in range(false_num):
        training_label.append(0)
    for i in range(true_num):
        training_label.append(1)

    # generate test set
    ADL_test = ADL_test[features]
    ADL_test = np.array(ADL_test).tolist()
    false_num = len(ADL_test)

    Fall_test = Fall_test[features]
    Fall_test = np.array(Fall_test).tolist()
    true_num = len(Fall_test)

    test_set = ADL_test + Fall_test
    test_label = []

    for i in range(false_num):
        test_label.append(0)
    for i in range(true_num):
        test_label.append(1)

    svc = svm.SVC()
    svc.fit(training_set, training_label)
    scores = svc.decision_function(test_set)
    fpr, tpr, thresholds = roc_curve(test_label, scores)

    label_pred = svc.predict(test_set)
    metric = confusion_matrix(test_label, label_pred)

    return fpr, tpr, metric


def confusion_metric_drawer(metric_array, plt_name):
    TP = metric_array[1][1]
    FP = metric_array[0][1]
    FN = metric_array[1][0]
    TN = metric_array[0][0]

    accuracy = (TP + TN) / (TP + TN + FP + FN)
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


fpr_thigh, tpr_thigh, metric_thigh = roc_drawer(normalized_mode="thigh_len")
fpr_torso, tpr_torso, metric_torso = roc_drawer(normalized_mode="torso_box")
fpr_none, tpr_none, metric_none = roc_drawer(normalized_mode="none")

# draw roc curve
plt.plot(fpr_thigh, tpr_thigh, color='cyan', label='thigh')
plt.plot(fpr_torso, tpr_torso, color='red', label='torso')
plt.plot(fpr_none, tpr_none, color='magenta', label='none')
plt.legend()
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.title('ROC curves for fall detection classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.savefig("../output/roc_curves.png")
plt.show()
plt.cla()

# draw confusion matrix
confusion_metric_drawer(metric_thigh, "../output/metric_thigh.png")
confusion_metric_drawer(metric_torso, "../output/metric_torso.png")
confusion_metric_drawer(metric_none, "../output/metric_none.png")
