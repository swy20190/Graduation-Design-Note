# this file classify data using decision tree model
import graphviz
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import os


os.environ["PATH"] += os.pathsep + 'G:/stable_windows_10_msbuild_Release_Win32_graphviz-2.46.0-win32/Graphviz/bin'
# generate the training set
ADL_data = pd.read_csv("../data/normalized/ADLNormAngle.csv", index_col=0)
ADL_training = ADL_data[['ratio', 'LAngle', 'RAngle']]
ADL_training = np.array(ADL_training)
ADL_training = ADL_training.tolist()
false_num = len(ADL_training)

Fall_data = pd.read_csv("../data/normalized/FallNormAngle.csv", index_col=0)
Fall_training = Fall_data[['ratio', 'LAngle', 'RAngle']]
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

# shuffle the data set
featureDf = pd.DataFrame(trainingSet, columns=['ratio', 'LAngle', 'RAngle'])
featureDf['Label'] = trainingLabel
shuffledDf = shuffle(featureDf)

# shuffled set
trainingSet = shuffledDf[['ratio', 'LAngle', 'RAngle']]
trainingLabel = shuffledDf['Label']

tree_RAngle_d3 = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
tree_RAngle_d3 = tree_RAngle_d3.fit(trainingSet, trainingLabel)

# 10-fold cross validation
scores_tree_RAngle_d3 = cross_val_score(tree_RAngle_d3, trainingSet, trainingLabel, cv=10)
print(scores_tree_RAngle_d3)

# visualize the decision tree
feature_name = ['aspect ratio', 'angle of left thigh', 'angle of right thigh']
class_name = ['ADL', 'Fall']
tree_RAngle_d3_dot = tree.export_graphviz(
    tree_RAngle_d3,
    out_file=None,
    feature_names=feature_name,
    class_names=class_name
)

graph = graphviz.Source(tree_RAngle_d3_dot)

graph.render("../output/TreeForRAngleD3")
