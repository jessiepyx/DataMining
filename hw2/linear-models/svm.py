import numpy as np
from scipy.optimize import minimize


def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    num = 0

    # YOUR CODE HERE

    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge with any method
    # that support constrain.

    # begin answer

    X_with_b = np.vstack((np.ones((1, N)), X))

    cons = ()
    for i in range(N):
        cons += ({
            'type': 'ineq',
            'fun': lambda t: min(y * np.matmul(t.T, X_with_b) - np.ones((1, N))),
        },)

    res = minimize(fun=lambda t: np.matmul(t.T, t) / 2,
                   jac=lambda t: t,
                   x0=w,
                   constraints=cons)

    w = np.transpose([res.x])

    dist = y * np.matmul(w.T, X_with_b)
    num = np.sum(abs(dist - 1) < 0.0001)

    # end answer
    return w, num
