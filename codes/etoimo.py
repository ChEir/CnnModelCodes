# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 23:56:11 2019

@author: Rena
"""

# two classes (class1, class2)
# only replace with a directory of yours
# finally .npy files saved in your directory with names train.npy, #test.npy, train_labels.npy, test_labels.npy
import cv2
import glob
import numpy as np
#Train data
train = []
train_labels = []
files = glob.glob ("C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/Step_1/P16-Deep-Learning-AZ/P16-Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset/training_set/cats/*.jpg") # your image path
for myFile in files:
    image = cv2.imread (myFile)
    train.append (image)
    train_labels.append([1., 0.])
files = glob.glob ("C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/Step_1/P16-Deep-Learning-AZ/P16-Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset/training_set/dogs/*.jpg")
for myFile in files:
    image = cv2.imread (myFile)
    train.append (image)
    train_labels.append([0., 1.])
    
    print(train.shape) #list at this point ,has no atrribute shape

train = np.array(train,dtype='float32') #as mnist
train_labels = np.array(train_labels,dtype='float64') #as mnist
# convert (number of images x height x width x number of channels) to (number of images x (height * width *3)) 
# for example (120 * 40 * 40 * 3)-> (120 * 4800)
train = np.reshape(train,[train.shape[0],train.shape[1]*train.shape[2]*train.shape[3]])

# save numpy array as .npy formats
np.save('train',train)
np.save('train_labels',train_labels)

#Test data
test = []
test_labels = []
files = glob.glob ("/data/test/class1/*.png")
for myFile in files:
    image = cv2.imread (myFile)
    test.append (image)
    test_labels.append([1., 0.]) # class1
files = glob.glob ("/data/test/class2/*.png")
for myFile in files:
    image = cv2.imread (myFile)
    test.append (image)
    test_labels.append([0., 1.]) # class2

test = np.array(test,dtype='float32') #as mnist example
test_labels = np.array(test_labels,dtype='float64') #as mnist
test = np.reshape(test,[test.shape[0],test.shape[1]*test.shape[2]*test.shape[3]])

# save numpy array as .npy formats
np.save('test',test) # saves test.npy
np.save('test_labels',test_labels)