import numpy as np


def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    # YOUR CODE HERE
    
    # begin answer

    max_iter = 1000  # samples may not be linearly separable, so a maximum iteration number is set

    X_with_b = np.vstack((np.ones((1, N)), X))  # make bias the first row of X

    # we first perform Cyclic Perceptron Learning Algorithm (assuming samples are linearly separable)
    # to get a classification that makes no mistake
    while True:
        finish = True

        for i in range(N):
            pred = np.dot(w.T, X_with_b[:, i])  # prediction

            if pred * y[:, i] <= 0:  # if predicts wrong
                finish = False
                w += np.transpose([X_with_b[:, i] * y[:, i]])  # update weight w.r.t. current sample

                iters += 1  # every time we update weight, it's another iteration
                if iters >= max_iter:
                    finish = True
                    break

        if finish:
            break

    # if max_iter is reached, the samples are probably not linearly separable
    # so we perform Pocket Algorithm and get as less mistakes as possible
    if iters == max_iter:
        # print("Too many iterations! Perform Pocket Algorithm instead.")

        # note that weights are already initialized by the previous PLA
        # w = np.random.random((P + 1, 1))  # randomly initialize pocket weights

        pocket_iter = 1000

        preds = np.sign(np.matmul(w.T, X_with_b)) * y  # predictions

        for i in range(pocket_iter):
            i += 1

            candidates = np.array(np.where(preds <= 0))  # false predictions

            if candidates.shape[1] == 0:  # if all right
                break

            np.random.shuffle(candidates)  # randomly choose one misclassified sample
            j = candidates[1, 0]

            # TRY updating weights w.r.t. current sample
            w_tmp = w + np.transpose([X_with_b[:, j] * y[:, j]])
            preds_tmp = np.sign(np.matmul(w_tmp.T, X_with_b)) * y

            # only when updated weights produce better classification, we actually update them
            # otherwise we don't update them
            if np.sum(preds_tmp) > np.sum(preds):
                w = w_tmp
                preds = preds_tmp
                # print("update")

        iters = pocket_iter

    # end answer
    
    return w, iters
