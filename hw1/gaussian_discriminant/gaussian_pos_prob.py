import numpy as np

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                within K Gaussian distributions.
    ''' 
    N = X.shape[1]
    K = Phi.shape[0]
    p = np.zeros((N, K))
    # Your code HERE

    # begin answer

    likelihood = np.zeros((N, K))
    for j in range(N):
        x = X[:, j]
        for i in range(K):
            Sigma0 = Sigma[:, :, i]
            Mu0 = x - Mu[:, i]
            likelihood[j, i] = (1 / 2 * np.pi * np.sqrt(np.linalg.det(Sigma0))\
                                * np.exp(-1/2 * Mu0.T.dot(np.linalg.inv(Sigma0)).dot(Mu0)))
    tmp = likelihood * Phi
    evidence = np.sum(tmp, axis=1)
    p = tmp / np.transpose([evidence])

    # end answer
    
    return p
