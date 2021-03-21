'''Tekne Consulting blogpost --- teknecons.com'''
'''pictures from https://www.kaggle.com/ivanfel/honey-bee-pollen'''


import numpy as np
import os
from PIL import Image

this_dir = os.path.dirname(os.path.abspath(__file__))
imag_dir = os.path.join(this_dir, 'images')


def imtoarr(file):
    pic = Image.open(os.path.join(imag_dir, file))
    arr = np.asarray(pic)
    return(arr)


pic_tensor = np.array([imtoarr(entry) for entry in os.scandir(imag_dir)])
labels = np.array([])


for entry in os.scandir(imag_dir):
    if os.path.basename(entry)[0] == 'P':
        labels = np.append(labels, 1)
    else:
        labels = np.append(labels, 0)


np.save(os.path.join(this_dir, 'bees_tensor.npy'), pic_tensor)
np.save(os.path.join(this_dir, 'labels.npy'), labels)
