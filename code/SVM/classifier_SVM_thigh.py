# This file classify the normalized data based on SVM
#  Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5, Left Elbow – 6,
#  Left Wrist – 7, Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, LAnkle – 13,
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt


def SVC_classifier(ADLDataFile, FallDataFile, features=None, k='rbf', C=1.0):
    """

    :param ADLDataFile: the normalized data of ADL
    :param FallDataFile: the normalized data of fall
    :param features: the key points chosen to train model
    :param k: kernel of svc
    :param C: the normalize index of svc
    :return: cross validation score
    """
    # default: nose, neck, shoulders, hips, knees, elbows
    if features is None:
        features = ['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '5x', '5y', '6x', '6y', '8x', '8y', '9x', '9y',
                    '11x', '11y', '12x', '12y']

    # generate data set
    ADL_data = pd.read_csv(ADLDataFile, index_col=0)
    ADL_training = ADL_data
    ADL_training = np.array(ADL_training)
    ADL_training = ADL_training.tolist()
    false_num = len(ADL_training)

    Fall_data = pd.read_csv(FallDataFile, index_col=0)
    Fall_training = Fall_data
    Fall_training = np.array(Fall_training)
    Fall_training = Fall_training.tolist()
    true_num = len(Fall_training)

    trainingSet = ADL_training + Fall_training

    # generate labels
    trainingLabel = []
    for i in range(false_num):
        trainingLabel.append(0)
    for i in range(true_num):
        trainingLabel.append(1)

    featureDf = pd.DataFrame(trainingSet, columns=['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '4x', '4y', '5x',
                                                   '5y', '6x', '6y', '7x', '7y', '8x', '8y', '9x', '9y', '10x', '10y',
                                                   '11x', '11y', '12x', '12y', '13x', '13y'])
    featureDf['Label'] = trainingLabel
    # shuffle
    shuffledTrainingDf = shuffle(featureDf)

    # select features
    trainingSet = shuffledTrainingDf[features]

    # convert label vector to one-dimension array
    trainingLabelVec = np.array(shuffledTrainingDf[['Label']]).tolist()
    trainingLabel = []
    for line in trainingLabelVec:
        trainingLabel.append(line[0])

    # SVM with rbf kernel
    svc_classifier = svm.SVC(C=C, kernel='rbf')
    svc_classifier.fit(trainingSet, trainingLabel)
    scores_svc_classifier = cross_val_score(svc_classifier, trainingSet, trainingLabel, cv=10)

    return scores_svc_classifier


scores_thigh = SVC_classifier("../data/normalized/ADLNormThigh.csv", "../data/normalized/FallNormThigh.csv")
print(scores_thigh)

# svc_thigh = svm.SVC()
#
#
# thigh_train, thigh_test, L_train, L_test = train_test_split(trainingSet, trainingLabel, test_size=.5)
# L_score = svc_thigh.fit(thigh_train, L_train).decision_function(thigh_test)
#
#
#
# fpr, tpr, thresholds = roc_curve(L_test, L_score)
#
# plt.plot(fpr, tpr)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.title('ROC curve for diabetes classifier')
# plt.xlabel('False Positive Rate (1 - Specificity)')
# plt.ylabel('True Positive Rate (Sensitivity)')
# plt.grid(True)
# plt.show()
