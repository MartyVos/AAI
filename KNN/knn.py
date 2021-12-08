from copy import deepcopy
import numpy as np
from typing import List


def get_dataset_labels():
    dates = np.genfromtxt("./KNN/dataset1.csv", delimiter=';', usecols=[0])
    labels = []
    for label in dates:
        if label < 20000301:
            labels.append("winter")
        elif 20000301 <= label < 20000601:
            labels.append("lente")
        elif 20000601 <= label < 20000901:
            labels.append("zomer")
        elif 20000901 <= label < 20001201:
            labels.append("herfst")
        else:
            labels.append("winter")
    return labels


def get_validation_labels():
    dates = np.genfromtxt("./KNN/validation1.csv", delimiter=';', usecols=[0])
    labels = []
    for label in dates:
        if label < 20010301:
            labels.append("winter")
        elif 20010301 <= label < 20010601:
            labels.append("lente")
        elif 20010601 <= label < 20010901:
            labels.append("zomer")
        elif 20010901 <= label < 20011201:
            labels.append("herfst")
        else:
            labels.append("winter")
    return labels


def normalize(array):
    # Xnorm = (X - minX) / (maxX - minX)
    min_x = min(array)
    max_x = max(array)
    for index in range(len(array)):
        array[index] = (array[index] - min_x) / (max_x - min_x)
    return array


def knn(datapoint, labels):
    pass

# Grab the values and rotate them 3x 90deg Left. First colom = firt row
dataset = np.rot90(np.genfromtxt("./KNN/dataset1.csv", delimiter=';'), 3)

norm_dataset = deepcopy(dataset)
for index in range(len(norm_dataset)):
    norm_dataset[index] = normalize(norm_dataset[index])

'''
Training set X of examples (~xi, yi) where
    – ~xi is feature vector of example i; and
    – yi is class label of example i.
Feature vector~x of test point that we want to classify.
Do:
    1. Compute distance D(~x,~xi);
    2. Select k closest instances x~j1,..., ~xjk with 
       class labels yj1,..., yjk
    3. Output class y∗, which is most frequent
       in yj1,..., yjk
'''