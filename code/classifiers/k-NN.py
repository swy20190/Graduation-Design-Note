# This file classify data using kNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# generate training set
ADL_training_df = pd.read_csv("../data/normalized/ADLNormThigh.csv")
ADL_training_list = np.array(ADL_training_df).tolist()
false_num = len(ADL_training_list)

Fall_training_df = pd.read_csv("../data/normalized/FallNormThigh.csv")
Fall_training_list = np.array(Fall_training_df).tolist()
true_num = len(Fall_training_list)

training_data = ADL_training_list + Fall_training_list

training_label = []
for i in range(false_num):
    training_label.append(0)
for i in range(true_num):
    training_label.append(1)

# generate test set
ADL_test_df = pd.read_csv("../data/normalized/ADLNormThighTest.csv")
ADL_test_list = np.array(ADL_test_df).tolist()
false_num = len(ADL_test_list)

Fall_test_df = pd.read_csv("../data/normalized/FallNormThighTest.csv")
Fall_test_list = np.array(Fall_test_df).tolist()
true_num = len(Fall_test_list)

test_data = ADL_test_list + Fall_test_list

test_label = []
for i in range(false_num):
    test_label.append(0)
for i in range(true_num):
    test_label.append(1)

# choose the best value of k
find_k_flag = input("Find the best k? (y/n)")
if find_k_flag == "y":
    k_range = range(1, 31)
    k_error = []
    for k in k_range:
        clf_knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf_knn, training_data, training_label, cv=6)
        k_error.append(1-scores.mean())

    plt.plot(k_range, k_error)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Error')
    plt.show()
