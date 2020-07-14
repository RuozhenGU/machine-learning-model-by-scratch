from GMM import EM
import numpy as np
from keras.datasets import mnist
from sklearn.decomposition import PCA

def fit_and_eval():

    k = 5

    # load data
    n_train = 60000
    n_test = 10000

    x_eval = np.zeros((n_test, 5))
    y_heta = np.zeros((n_test, 1))

    size = 28 # size of image is 28 pixels

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # flat the image
    x_train = x_train.reshape((n_train, size ** 2)) / 255
    x_test = x_test.reshape((n_test, size ** 2)) / 255

    # divide into 10 classes
    classes = [[] for _ in range(10)]

    for _x, _y in zip(x_train, y_train):
        classes[_y].append(_x)

    # PCA to reduce dimensions
    pca_clf = PCA(n_components=50)
    pca_clf.fit(x_train)

    classes_after_pca = [pca_clf.transform(classes[i]) for i in range(10)]
    test_after_pca = pca_clf.transform(x_test)

    # training for each class
    train_result = [EM(classes_after_pca[i], k) for i in range(10)]

    # evaluation on test data
    for i in range(10):
        means, Sk, pi_k, loss = train_result[i]

        for _k in range(k):
            mean_diff = test_after_pca - means[_k]
            expo = np.exp(-0.5 * np.einsum('ij,ij->i', np.divide(mean_diff, Sk[_k]), mean_diff))
            x_eval[:, _k] = (1./np.sqrt(np.prod(Sk[_k]))) * pi_k[_k] * expo
        
        y_heta = np.c_[y_heta, np.sum(x_eval, axis=1)]

    y_heta = y_heta[:, 1:]

    # pick the final prediction (which distribution) by selecting the largest prob
    pred = [list(y_heta[_i, :]).index(max(list(y_heta[_i, :]))) for _i in range(n_test)]

    error = [int(pred[i] != y_test[i]) for i in range(len(pred))]

    print("error rate is: %f" % float(sum(error) / len(pred)))


if __name__ == '__main__':
    fit_and_eval()







