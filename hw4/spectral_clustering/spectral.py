import numpy as np
from kmeans import kmeans


def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    # begin answer

    N = W.shape[0]

    # diagonal matrix
    D = np.diag(np.sum(W, axis=0))

    # graph laplacian
    L = D - W

    # eigenvalues and eigenvectors
    val, vec = np.linalg.eigh(L)

    # k-means
    idx = kmeans(vec[:, 0:k], k)

    return idx

    # end answer
