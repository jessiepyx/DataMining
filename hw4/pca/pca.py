import numpy as np


def PCA(data):
    '''
    PCA	Principal Component Analysis

    Input:
      data      - Data numpy array. Each row vector of fea is a data point.
    Output:
      eigvector - Each column is an embedding function, for a new
                  data point (row vector) x,  y = x*eigvector
                  will be the embedding result of x.
      eigvalue  - The sorted eigvalue of PCA eigen-problem.
    '''

    # YOUR CODE HERE
    # Hint: you may need to normalize the data before applying PCA
    # begin answer

    data_normed = data - np.mean(data, axis=0).reshape(1, -1)
    val, vec = np.linalg.eigh(np.cov(data_normed.T))
    return vec.T[::-1].T, val[::-1]  # sort from largest to smallest

    # end answer
