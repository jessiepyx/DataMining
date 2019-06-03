import numpy as np
import random

def kmeans(x, k):
    '''
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
                ctrs - cluster centers, K-by-p matrix.
                iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    '''
    # YOUR CODE HERE

    # begin answer
    n, p = x.shape
    iter_ctrs = []
    epsilon = 0.1
    idx = np.zeros((n, )).astype(int)

    ctrs = np.array(random.choices(x, k=k))
    iter_ctrs.append(np.copy(ctrs))

    # assign labels
    for i in range(n):
        dist = np.linalg.norm((x[i]).reshape(1, p) - ctrs, axis=1)
        idx[i] = np.argmin(dist)

    # update centers
    for i in range(k):
        ctrs[i] = np.mean(x[idx == i, :], axis=0)

    while np.linalg.norm(ctrs - iter_ctrs[-1]) > epsilon:
        iter_ctrs.append(np.copy(ctrs))

        # assign labels
        for i in range(n):
            dist = np.linalg.norm((x[i]).reshape(1, p) - ctrs, axis=1)
            idx[i] = np.argmin(dist)

        # update centers
        for i in range(k):
            ctrs[i] = np.mean(x[idx == i, :], axis=0)

    iter_ctrs.append(np.copy(ctrs))
    iter_ctrs = np.array(iter_ctrs)

    # end answer

    return idx, ctrs, iter_ctrs
