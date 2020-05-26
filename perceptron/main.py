import pandas as pd
from perceptron import Perceptron

def load_data(x_path='./spambase_X.csv', y_path='./spambase_Y.csv'):
    X = pd.read_csv(x_path, header= None)
    Y = pd.read_csv(y_path, header = None)

    # transpose & truncate
    X = (X.T)
    
    return X, Y
    

def init():

    # load data
    X, Y = load_data()

    # init model
    perceptron = Perceptron()

    # train
    wt_matrix, b_vecb_vec = perceptron.train(X, Y, 500)

    # tuned w, b
    w = perceptron.w
    b = perceptron.b
    mistake = perceptron.mistake

    # plot
    perceptron.plot()


if __name__ == "__main__":
    init()
