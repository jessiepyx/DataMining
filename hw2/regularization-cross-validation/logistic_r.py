import numpy as np


def logistic_r(X, y, lmbda):
    '''
    LR Logistic Regression.

      INPUT:  X:   training sample features, P-by-N matrix.
              y:   training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))

    # YOUR CODE HERE
    # begin answer

    X_with_b = np.vstack((np.ones((1, N)), X))

    min_loss = 0
    max_iters = 1000
    learning_rate = 10

    iter = 0
    h = 1 / (1 + np.exp(-w.T.dot(X_with_b)))  # sigmoid
    loss = - y.dot(np.log(h).T) + (1 - y).dot(np.log(1 - h).T) / N  # average loss of N samples
    loss += lmbda * w.T.dot(w) / 2 / N  # L-2 regularization

    while loss[0, 0] > min_loss and iter < max_iters:
        gradient = - X_with_b.dot((y - h).T) / N
        gradient += lmbda * w / N
        w = w - learning_rate * gradient  # gradient descent

        # update
        iter += 1
        h = 1 / (1 + np.exp(-w.T.dot(X_with_b)))
        loss = - y.dot(np.log(h).T) + (1 - y).dot(np.log(1 - h).T) / N
        loss += lmbda * w.T.dot(w) / 2 / N

    # end answer
    
    return w
