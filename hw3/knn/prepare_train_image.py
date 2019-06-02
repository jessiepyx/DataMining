from extract_image import extract_image
from show_image import show_image
import numpy as np

x_train = np.zeros((5 * 100, 140))
y_train = np.zeros((5 * 100, ))

for i in range(20):
    filename = 'captcha_train_data/' + str(i) + '.jpg'
    x_train[5*i: 5*i+5] = extract_image(filename)
    show_image(x_train[5*i: 5*i+5])
    label = input('label ' + str(i))
    for j in range(5):
        y_train[5*i+j] = eval(label[j])
    print(y_train[5*i: 5*i+5])

np.savez('hack_data.npz', x_train=x_train, y_train=y_train)
