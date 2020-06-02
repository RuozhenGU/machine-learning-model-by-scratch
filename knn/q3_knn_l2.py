import random
import numpy as np
import sys
import math    
from q1_gen_data import data_gen
np.random.seed(0)

def get_neighbors(dist, X, y, k, d, i_origin):
    
    distances = [(i, dist(y, X[i], d)) for i in range(len(X)) if i != i_origin]
    distances.sort(key=lambda pair: pair[1])
    neighbours = [distances[i][0] for i in range(k)]

    return neighbours


def predict(neighbours, y):
    votes = {1: 0, -1: 0} # assume binary classification only
    for index in neighbours:
        vote = y[index]
        votes[vote] += 1
    if votes[1] > votes[-1]:
        return 1
    else:
        return -1


def pred_error(preds, y):
    error = 0
    for i in range(len(y)):
        if preds[i] != y[i]:
            error += 1
    return error / len(y)

    
def euclidean_distance_l2(point1, point2, d):
    distance = 0.
    for i in range(d - 1):
        distance += (point1[i] - point2[i])**2
    return math.sqrt(distance)
      

def knn(trainX, trainY, testX, k_lst, dist, testY):
    n, d = trainX.shape
    error = 0
    for k in k_lst:
        error = 0
        preds = []
        for i in range(n):
            neighbours = get_neighbors(dist,trainX, testX[i], k, d, i)
            preds.append(predict(neighbours, trainY))
        accuracy = pred_error(preds, testY)
        print("k = %d, accuracy = %f" % (k, accuracy))
        
    return preds, accuracy


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
    testYhat, error = knn(trainX, trainY, testX, k_lst, euclidean_distance_l2, testY)
    
            
        

