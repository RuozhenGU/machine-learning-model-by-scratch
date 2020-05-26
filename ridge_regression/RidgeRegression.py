import numpy as np
import pandas as pd
from numpy import linalg


class RidgeRegression:

    """
        Ridge Regression using Gradient Descent
    """

    def __init__(self, max_pass=4000, _lambda=1, lr=0.00000001, tol=1e-9, closed_form=False, normalize=False):
        self.train_loss = np.zeros(max_pass) 
        self.test_error = np.zeros(max_pass)
        self.train_error = np.zeros(max_pass)
        self.max_pass = max_pass
        self._lambda = _lambda
        self.lr = lr
        self.tol = tol
        self.w = []
        self.b = []
        self.closed_form = closed_form
        self.normalize = normalize

    def cost_function(self, w, b, X, y):
        """
            compute loss using ridge regressor
        """
        
        n, _ = X.shape        
        
        J = (1. / (2 * n)) * ((linalg.norm(X @ w - y))**2) \
            + self._lambda * (linalg.norm(w))**2
     
        return J
    
    def error_function(self, w, b, X, y):
        """
            compute error using ridge regressor
        """
        
        n, _ = X.shape        
        
        E = (1. / (2 * n)) * ((linalg.norm(X @ w - y))**2)
            
     
        return E
    
    def closed_form_b(self, X, y, w): 
        '''Closed form solution of b for ridge regression'''
        n, _ = X.shape
        return (1. / n) * (y - X @ w)


    @staticmethod
    def normalize(X, y, X_test, y_test):
        ''' normalize input by subtract mean'''
        X -= np.mean(X, axis=0)
        X_test -= np.mean(X_test, axis=0)
        return X, X_test
        


    def gradient_descent(self, X, y, X_test, y_test, w, b):
        '''Gradient descent for ridge regression'''

        #Initialisation of useful values 
        n, _ = X.shape
        

        for i in range(self.max_pass):
            
            # compute current loss
            self.train_loss[i] = self.cost_function(w, b, X, y)
            self.train_error[i] = self.error_function(w, b, X, y)
            self.test_error[i] = self.error_function(w, b, X_test, y_test)
            
            print("### epoch %d - loss: %d" % (i, self.train_loss[i]))
            
            # update 
            gradient = (1. / n) * (X.T @(X @ w + b - y)) + 2 * self._lambda * w
            
            w = w - self.lr * gradient
            
            if self.closed_form:
                b = self.closed_form_b(X, y, w)
            else:
                b = b - (self.lr / n) * (X @ w + b - y)
           
            # gradient vanish
            if len(self.w) > 0 and linalg.norm((self.w)[-1] - w) <= self.tol:
                print("no long descent at epoch=%d" % i+1)
                break

            # save
            (self.w).append(w)
            (self.b).append(b)

        return w, b


    def fit(self, X, y, X_test, y_test):
        
        # init hyperparameter
        w0 = np.zeros((X.shape[1], 1))
        b0 = 0.
        
        # normalize X
        if self.normalize:
            X, X_test = normalize(self, X, y, X_test, y_test)
            
        # fit the model
        w, b = self.gradient_descent(X, y, X_test, y_test, w0, b0)

        return w, b
        