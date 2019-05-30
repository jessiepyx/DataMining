import numpy as np


def ridge(X, y, lmbda):
    '''
    RIDGE Ridge Regression.

      INPUT:  X: training sample features, P-by-N matrix.
              y: training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w: learned parameters, (P+1)-by-1 column vector.

    NOTE: You can use pinv() if the matrix is singular.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer

    X_with_b = np.vstack((np.ones((1, N)), X))
    
    mat = np.matmul(X_with_b, X_with_b.T) + lmbda * np.eye(P + 1)
    
    if lmbda == 0:
        w = np.linalg.pinv(mat)
    else:
        w = np.linalg.inv(mat)

    w = w.dot(X_with_b).dot(y.T)

    # end answer
    return w
