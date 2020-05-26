import pandas as pd
from matplotlib import pyplot as plt
from RidgeRegression import RidgeRegression


def load_data(path='./'):

    housing_X_test = pd.read_csv(path + "housing_X_test.csv", header=None)
    housing_X_train = pd.read_csv(path + "housing_X_train.csv", header=None)
    housing_y_test = pd.read_csv(path + "housing_y_test.csv", header=None)
    housing_y_train = pd.read_csv(path + "housing_y_train.csv", header=None)

    return housing_X_test.T, housing_X_train.T, housing_y_test, housing_y_train


def plot(train_loss, test_error, train_error):
    plt.figure(figsize=(15,8))
    plt.plot(train_loss, label="train loss")
    plt.plot(test_error, label="test error")
    plt.plot(train_error, label="train error")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

def main(path='./', plot=True):

    housing_X_test, housing_X_train, housing_y_test, housing_y_train = load_data(path)

    clf = RidgeRegression(max_pass=200000, 
                          _lambda=10, 
                          lr=0.00000001, 
                          tol=1e-9,
                          closed_form=True,
                          normalize=True)

    clf.fit(housing_X_train, housing_y_train, housing_X_test, housing_y_test)

    if plot:

        train_loss = clf.train_loss
        test_error = clf.test_error
        train_error = clf.train_error

        plot(train_loss, test_error, train_error)


if __name__ == "__main__":
    main()