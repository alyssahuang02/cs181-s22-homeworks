#####################
# CS 181, Spring 2022
# Homework 1, Problem 2
# Start Code
##################

import math
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# set up data
data = [(0., 0.),
        (1., 0.5),
        (2., 1),
        (3., 2),
        (4., 1),
        (6., 1.5),
        (8., 0.5)]

x_train = np.array([d[0] for d in data])
y_train = np.array([d[1] for d in data])

x_test = np.arange(0, 12, .1)

print("y is:")
print(y_train)

def kern(x, x_prime, tau):
    return np.exp(-1*np.transpose(x - x_prime)*(x-x_prime)/tau)

def predict_knn(k=1, tau=1):
    """Returns predictions for the values in x_test, using KNN predictor with the specified k."""
    res = np.zeros(len(x_test))
    index = 0
    for x in x_test:
        arr = []
        for i in range(len(data)):
            x_t = x_train[i]
            arr.append((kern(x, x_t, tau), i))
        arr.sort(reverse=True)
        sum = 0
        for i in range(k):
            print(arr)
            sum += y_train[int(arr[i][1])]
        res[index] = sum/k
        index += 1

    return res


def plot_knn_preds(k):
    plt.xlim([0, 12])
    plt.ylim([0,3])
    
    y_test = predict_knn(k=k)
    
    plt.scatter(x_train, y_train, label = "training data", color = 'black')
    plt.plot(x_test, y_test, label = "predictions using k = " + str(k))

    plt.legend()
    plt.title("KNN Predictions with k = " + str(k))
    plt.savefig('k' + str(k) + '.png')
    plt.show()

for k in (1, 3, len(x_train)-1):
    plot_knn_preds(k)