from urllib.request import urlretrieve
from extract_image import extract_image
from show_image import show_image
import numpy as np

url = 'http://jwbinfosys.zju.edu.cn/CheckCode.aspx'

for i in range(20):
    filename = 'captcha_train_data/' + str(i) + '.jpg'
    urlretrieve(url, filename)

filename = 'CheckCode.aspx'
urlretrieve(url, filename)
