import numpy as np

def knn_graph(X, k, threshold):
    '''
    KNN_GRAPH Construct W using KNN graph

        Input:
            X - data point features, n-by-p maxtirx.
            k - number of nn.
            threshold - distance threshold.

        Output:
            W - adjacency matrix, n-by-n matrix.
    '''

    # YOUR CODE HERE
    # begin answer

    N, P = X.shape

    dist = np.diag([float("inf")] * N)
    W = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(X[i] - X[j])
            dist[i][j] = d
            dist[j][i] = d

    idx = np.argsort(dist, axis=1)

    for i in range(N):
        W[i, idx[i, 0:k]] = dist[i, idx[i, 0:k]]

    W[W > threshold] = 0
    W[W > 0] = 1

    return W

    # end answer
