# This file classify the NormThigh data based on SVM
#  Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5, Left Elbow – 6,
#  Left Wrist – 7, Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, LAnkle – 13,
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle


ADL_data = pd.read_csv("../data/normalized/ADLNormThigh.csv", index_col=0)
# all the features are taken into consideration
ADL_training = ADL_data
ADL_training = np.array(ADL_training)
ADL_training = ADL_training.tolist()
false_num = len(ADL_training)

Fall_data = pd.read_csv("../data/normalized/FallNormThigh.csv", index_col=0)
Fall_training = Fall_data
Fall_training = np.array(Fall_training)
Fall_training = Fall_training.tolist()
true_num = len(Fall_training)

trainingSet = ADL_training + Fall_training

# generate the training label
trainingLabel = []

for i in range(false_num):
    trainingLabel.append(0)
for i in range(true_num):
    trainingLabel.append(1)

# shuffle the dataset
featureDf = pd.DataFrame(trainingSet, columns=['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '4x', '4y', '5x', '5y',
                                               '6x', '6y', '7x', '7y', '8x', '8y', '9x', '9y', '10x', '10y', '11x',
                                               '11y', '12x', '12y', '13x', '13y'])

featureDf['Label'] = trainingLabel
shuffledTrainingDf = shuffle(featureDf)


# shuffled set
trainingSet = shuffledTrainingDf[['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '4x', '4y', '5x', '5y', '6x', '6y',
                                 '7x', '7y', '8x', '8y', '9x', '9y', '10x', '10y', '11x', '11y', '12x', '12y', '13x',
                                  '13y']]

trainingLabelVec = np.array(shuffledTrainingDf[['Label']]).tolist()
trainingLabel = []
for line in trainingLabelVec:
    trainingLabel.append(line[0])

# SVM with RBF kernel
for i in range(10):
    svc_thigh = svm.SVC(C=0.1*(i+1), kernel='rbf')
    svc_thigh.fit(trainingSet, trainingLabel)
    scores_svc_thigh = cross_val_score(svc_thigh, trainingSet, trainingLabel, cv=10)
    print("SVC RBF kernel C=" + str(0.1*(i+1)) + ":\n")
    print(scores_svc_thigh)
