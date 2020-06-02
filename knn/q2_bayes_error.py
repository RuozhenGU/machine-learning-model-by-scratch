from q1_gen_data import data_gen
import random
import numpy as np
import sys
import math    
np.random.seed(0)


def Bayes(X, y, w, b):
    n, d = X.shape
    total_error = 0
    for i in range(n):
        Z = (np.dot(w.T, X[i]) + b)
        pred_y = 1 if Z >= 0 else -1
        if pred_y != y[i]:
            total_error += 1
    return total_error / n


if __name__ == "__main__":
    n = 1000000
    d = 2
    w = np.zeros((d,))
    w[0] = 1
    b = 0
    sigma = 1

    X, y = data_gen(n,d,sigma,w,0)
    error = Bayes(X, y, w, 0)
    print("Bayes Error is simulated using %d data as %f" % (n, error))