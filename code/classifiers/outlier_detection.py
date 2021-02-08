# This file is based on outlier detection method
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from novelty_detection import confusion_metric_drawer_novelty


# generate training set
ADL_train_df = pd.read_csv("../data/normalized/ADLNormThigh.csv")
ADL_train_df = ADL_train_df[['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '5x', '5y', '6x', '6y', '8x', '8y', '9x',
                             '9y', '11x', '11y', '12x', '12y']]
ADL_train = np.array(ADL_train_df).tolist()

Fall_train_df = pd.read_csv("../data/normalized/FallNormThigh.csv")
Fall_train_df = Fall_train_df[['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '5x', '5y', '6x', '6y', '8x', '8y', '9x',
                               '9y', '11x', '11y', '12x', '12y']]
Fall_train = np.array(Fall_train_df).tolist()

training_set = ADL_train + Fall_train

# generate test set
ADL_test_df = pd.read_csv("../data/normalized/ADLNormThighTest.csv")
ADL_test_df = ADL_test_df[['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '5x', '5y', '6x', '6y', '8x', '8y', '9x',
                           '9y', '11x', '11y', '12x', '12y']]
ADL_test = np.array(ADL_test_df).tolist()
num_inner = len(ADL_test)

Fall_test_df = pd.read_csv("../data/normalized/FallNormThighTest.csv")
Fall_test_df = Fall_test_df[['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '5x', '5y', '6x', '6y', '8x', '8y', '9x',
                             '9y', '11x', '11y', '12x', '12y']]
Fall_test = np.array(Fall_test_df).tolist()
num_outlier = len(Fall_test)

test_set = ADL_test + Fall_test

test_label = []
for i in range(num_inner):
    test_label.append(1)
for i in range(num_outlier):
    test_label.append(-1)

# Isolation tree
clf_ITree = IsolationForest(max_samples=512, max_features=10)
clf_ITree.fit(training_set)
predict_label = clf_ITree.predict(test_set)
metric_ITree = confusion_matrix(test_label, predict_label)
confusion_metric_drawer_novelty(metric_ITree, "../output/metric_thigh_ITree.png")

# Fitting an elliptic envelope
clf_EEnvelope = EllipticEnvelope(contamination=0.05)
clf_EEnvelope.fit(training_set)
predict_label = clf_EEnvelope.predict(test_set)
metric_EEnvelope = confusion_matrix(test_label, predict_label)
confusion_metric_drawer_novelty(metric_EEnvelope, "../output/metric_thigh_EEnvelope.png")

# local outlier factor
clf_LOF = LocalOutlierFactor(contamination=0.12)
predict_label = clf_LOF.fit_predict(test_set)
metric_LOF = confusion_matrix(test_label, predict_label)
confusion_metric_drawer_novelty(metric_LOF, "../output/metric_thigh_LOF_o.png")
