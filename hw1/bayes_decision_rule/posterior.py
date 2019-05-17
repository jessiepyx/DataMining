import numpy as np
from likelihood import likelihood

def posterior(x):
    '''
    POSTERIOR Two Class Posterior Using Bayes Formula
    INPUT:  x, features of different class, C-By-N vector
            C is the number of classes, N is the number of different feature
    OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
    '''

    C, N = x.shape
    l = likelihood(x)
    total = np.sum(x)
    p = np.zeros((C, N))
    #TODO

    # begin answer

    prior = np.sum(x, axis=1) / total
    evidence = np.sum(x, axis=0) / total
    p = l * prior.reshape(C, 1) / evidence.reshape(1, N)

    # end answer
    
    return p
