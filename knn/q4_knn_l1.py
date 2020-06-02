import random
import numpy as np
import sys
import math    
from q1_gen_data import data_gen
from q3_knn_l2 import knn
np.random.seed(0)


def euclidean_distance_l1(point1, point2, d):
    distance = 0.
    for i in range(d - 1):
        distance += np.abs(point1[i] - point2[i])
    return math.sqrt(distance)


if __name__ == "__main__":

    # init params
    d = 2
    w = np.zeros((d,))
    w[0] = 1
    b = 0
    sigma = 1

    # define k
    k_lst = [1, 3, 5]

    # generate train and test dataset
    trainX, trainY = data_gen(6000, 2, sigma, w, 0)
    testX, testY = data_gen(6000, 2, sigma, w, 0)

    # fit model
    testYhat, error = knn(trainX, trainY, testX, k_lst, euclidean_distance_l1, testY)
    