import numpy as np
import matplotlib.pyplot as plt
from pca import PCA


def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    img_r = (plt.imread(filename)).astype(np.float64)

    # YOUR CODE HERE
    # begin answer

    # pixels with alpha > 0
    pixel = np.array(np.where(img_r[:, :, -1] > 0)).T

    eigenvec, eigenval = PCA(pixel)

    embedding = np.matmul(pixel, eigenvec[:, :2]).astype(int)
    x = -embedding[:, 1]
    y = embedding[:, 0]
    x = x - min(x)
    y = y - min(y)

    img = np.zeros((max(x) + 1, max(y) + 1, img_r.shape[-1] - 1))
    for i in range(embedding.shape[0]):
        img[x[i], y[i]] = img_r[pixel[i, 0], pixel[i, 1], :-1]
    img = np.array(img).astype(int)

    return img

    # end answer
