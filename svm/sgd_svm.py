import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    


class SGDSVM:

    """
        Support Vector Machine classifier using Stochastic Gradient Descent and 
        squared hinge loss function.
    """

    def __init__(self, features, num_data, C, batch_size=1, lr=0.000001, epochs=500):
        self.w = np.zeros(features)
        self.b = 0.
        self.lr = lr
        self.epochs = epochs
        self.loss = np.zeros((num_data, ))
        self.indices = np.arange(num_data)
        self.batch_size = batch_size
        self.C = C


    def select_batch(self, X, y):

        def select_random_index(self):
  
            return np.random.choice(self.indices, self.batch_size, replace=False) 
        
        idx = self.select_random_index()
        X_batch = np.take(X, idx, 0)
        y_batch = y[idx]
        return X_batch, y_batch  

    def squared_hinge_loss(self, X, y):

        n = X.shape[0]
        
        for i, x in enumerate(X):
            wx = np.dot(x, self.w)
            z = (1 - y[i] * (wx + self.b)) ** 2
            loss[idx] = max(z, 0)  
        return loss

    def activate(self, x, y):
        return 1 if y * ((self.w @ x) + self.b) <= 1 else 0

    def fit(self, X, Y):
        
        # training
        for i in range(self.epochs):
            for i, x in enumerate(X):
                y = Y[i]
                # activate 
                
                if (self.activate(x, y[0])):
                    y_head = np.dot(x, self.w) + self.b
                    self.w += self.lr * 2 * (self.C * (1 - y * y_head) * y) * x
                    self.b += self.lr * 2 * self.C * (1 - y * y_head) * y
                
                self.w = self.w / (1 + self.lr)
                # TODO: show loss here by calling squared_hinge_loss(X, Y)
        return self.w, self.b


def train():
    # create naive fake data points
    X = np.array([[1, 2],[2, 1],[3, 1],[3, 2]])
    y = np.array([[1.],[1.],[-1.],[-1.]])    
    
    n, d = X.shape

    clf = SGDSVM(d, n, 86131)        
    w, b = clf.fit(X, y)
    print(w)
    print(b)

if __name__ == "__main__":
    train()