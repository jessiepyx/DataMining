import numpy as np


def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))

    # YOUR CODE HERE
    # begin answer

    X_with_b = np.vstack((np.ones((1, N)), X))

    min_loss = 0
    max_iters = 10000
    learning_rate = 0.5
    decay_rate = 0.98
    decay_step = 1000

    iter = 0
    h = 1 / (1 + np.exp(-w.T.dot(X_with_b)))  # sigmoid
    loss = y.dot(np.log(h).T) + (1 - y).dot(np.log(1 - h).T) / N  # average loss of N samples

    while -loss[0, 0] > min_loss and iter < max_iters:
        gradient = -X_with_b.dot((y - h).T) / N
        w = w - learning_rate * gradient  # gradient descent

        # update
        iter += 1
        h = 1 / (1 + np.exp(-w.T.dot(X_with_b)))
        loss = y.dot(np.log(h).T) + (1 - y).dot(np.log(1 - h).T) / N
        learning_rate = learning_rate * (decay_rate ** (iter / decay_step))  # exponential decay

    # end answer
    
    return w
