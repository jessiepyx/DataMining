import numpy as np

def linear_regression(X, y):
    '''
    LINEAR_REGRESSION Linear Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer

    X_with_b = np.vstack((np.ones((1, X.shape[1])), X))
    w = np.linalg.pinv(X_with_b.dot(X_with_b.T)).dot(X_with_b).dot(y.T)

    # end answer
    return w
