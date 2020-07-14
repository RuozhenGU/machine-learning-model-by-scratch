import numpy as np


def EM(X, k, maxiter=500, tol=1e-9):

    n, d = X.shape
    means = []
    Sk = []
    loss = []
    ll_latest = None
    ll_prev = None
    
    # Initialize r as nxk
    resp = np.zeros((n, k))
    resp[:, k-1] = 1

    # Initialize mean 
    X_index = np.arange(n)

    np.random.shuffle(np.arange(n))

    X_shuffled = np.array([X[i] for i in X_index])

    clusters_size = int(np.floor(n / k))
    X_split = [X_shuffled[i:i + clusters_size] for i in range(0, n, clusters_size)]

    for i in np.arange(k):
        means.append(np.mean(X_split[i], axis=0))
    
    means = np.array(means)

    # initialize covariance
    np.apply_along_axis(np.random.shuffle, 1, resp)
    Sk = (np.matmul((X ** 2).T, resp) / np.sum(resp, axis=0)).T - (means ** 2)
    
    
    for i in range(maxiter):

        pi_k = np.sum(resp, axis=0) / n

        # E step: compute responsibilities
        for _k in range(k):
            mean_diff = X - means[_k]
            expo = np.exp(-0.5 * np.einsum('ij,ij->i', np.divide(mean_diff, Sk[_k]), mean_diff))
     
            resp[:, _k] = (1./np.sqrt(np.prod(Sk[_k]))) * pi_k[_k] * expo

        # normalize responsibility
        r_i = resp.sum(axis=1)[:, np.newaxis]
        resp = resp / r_i

        # compute loss
        if ll_prev:
            ll_prev = ll_latest
            ll_latest = 0. - np.sum(np.log(r_i * 2 * np.pi), axis=0)
        else:
            ll_latest = 0. - np.sum(np.log(r_i * 2 * np.pi), axis=0)
            ll_prev = ll_latest
        
        loss.append(ll_latest)
        if i % 5 == 0:
            print("# loss: %f" % ll_latest)

        # Check for convergence
        if abs(ll_latest - ll_prev) <= tol * abs(ll_latest) and i > 0:
            print("Done", i)
            break
        

        # update
        means = (np.matmul(X.T, resp) / np.sum(resp, axis=0)).T

        Sk = (np.matmul((X ** 2).T, resp) / np.sum(resp, axis=0)).T - (means ** 2)

    return means, Sk, pi_k, loss