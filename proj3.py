import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

from PIL import Image
from tkinter import Tcl
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

# Load list of labels/classes
with open('./data/cifar-10-binary/cifar-10-batches-bin/batches.meta.txt', 'r') as f:
    labels = [line.rstrip() for line in f]

print("=========== Classes ===========")
for i in range(len(labels)-1): print(i,":", labels[i])


import matplotlib.pyplot as plt

# image shape
HEIGHT = 32
WIDTH = 32
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

def read_all_images(b):
    images = np.reshape(b, (-1, 3, 32, 32))
    images = np.transpose(images, (0, 3, 2, 1))
    return images

def read_single_image(image_file):
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    #image = np.array(image_file, dtype=np.uint8, count=SIZE)
    image = np.reshape(image, (3, 32, 32))
    image = np.transpose(image, (2, 1, 0))
    return image

def plot_image(image):
    plt.imshow(image)
    plt.show()

image = read_single_image()
