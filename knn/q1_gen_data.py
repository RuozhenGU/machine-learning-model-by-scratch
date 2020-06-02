import random
import numpy as np
import sys
import math    
np.random.seed(0)


def data_gen(n, d, sigma, w, b):
    
    def sigmoid(x):
      return 1 / (1 + np.exp(-x))
    
    dataset = []
    labels = []
    
    for iter in range(n):
        X = (np.random.random(d) * 2 - 1)
        U = random.uniform(0, 1)
        
        Y = 1 if U <= sigmoid((np.dot(w.T, X) + b) / sigma) else -1
        dataset.extend(X.reshape(1, 2))
        labels.extend([Y])
        
    return np.array(dataset), np.array(labels)




if __name__ == "__main__":
    n = 3000
    d = 2
    w = np.zeros((d,))
    w[0] = 1
    b = 0
    sigma = 1

    X, y = data_gen(n,d,sigma,w,0)
