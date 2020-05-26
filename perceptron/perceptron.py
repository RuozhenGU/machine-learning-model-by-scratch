import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self):
        self.w = None
        self.b = None
        self.mistakes = []
        self.mistake_count = 0

    def activate(self, x, y):
        return 1 if np.dot(y, (np.dot(self.w, x) + self.b)) <= 0 else 0

    def train(self, X, Y, epochs=500):

        train_count = 0
        n, d = X.shape
        wt_matrix = []
        b_vec = []

        # init parameters
        self.w = np.zeros(d)
        self.b = 0
        # training
        for i in range(epochs):
            for i in range(n):
                x = X.iloc[i]
                y = Y.iloc[i]
                # activate 
                if (self.activate(x, y[0])):
                    self.w += y[0] * x
                    self.b += y[0]
                    self.mistake_count += 1
                
            print(self.mistake_count)
            self.mistakes.append(self.mistake_count)
            self.mistake_count = 0 

            # store w, b    
            wt_matrix.append(self.w) 
            b_vec.append(self.b)   
            
        return wt_matrix[-1], b_vec[-1]


    def plot(self):
        plt.plot(self.mistakes)
        plt.xlabel("Passes #")
        plt.ylabel("Mistake Count")
        plt.show()