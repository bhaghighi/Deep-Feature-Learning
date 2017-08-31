#%%
# Handwritten Digit Recognition using CNN in python with keras 
# http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
import keras 
#%%
from keras.datasets import mnist
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

#%%
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot the first image of training set
for i in range(10):
    plt.figure()
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))