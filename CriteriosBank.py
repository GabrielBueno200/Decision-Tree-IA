import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from scipy.io import arff

def bankCriteria():
    data, meta = arff.loadarff('./bank.arff')

    # region Defining the tree
    attributes = ['age','average', 'day','duration','campaign','pdays','previous']
    attributesValues = [np.asarray(data[attribute]).reshape(-1, 1) for attribute in attributes]

    features = np.concatenate(attributesValues, axis=1)
    target = data['subscribed']
    # endregion

    decisionTree = DecisionTreeClassifier(criterion='entropy').fit(features, target)

    # region Plot
    plt.figure(figsize=(10, 6.5))
    tree.plot_tree(decisionTree, feature_names=attributes, class_names=['yes', 'no'], filled=True, rounded=True)
    plt.show()

    fig, ax = plt.subplots(figsize=(25, 10))
    metrics.plot_confusion_matrix(decisionTree, features, target, display_labels=['yes', 'no'], values_format='d',ax=ax)
    plt.show()
    # endregion

bankCriteria()