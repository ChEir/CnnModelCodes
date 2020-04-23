# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:58:34 2019

@author: Rena
"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
from keras.preprocessing import image
from keras.layers import Dropout
import os
import cv2
from pathlib import Path 
from PIL import Image
import csv
import glob
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
#E:\Πτυχιακη\Step2\Data\H_Channel_endoscopic_data
#E:/Πτυχιακη/Step_1/HSV_endoscopic_data/train/normal/
disk_dir= Path('E:/Πτυχιακη/Step2/Data/H_Channel_endoscopic_data/test/normal/')

#has to give it HSV data 
X_data = [] 
#C:/Users/Rena/Desktop/normal/*.jpg
files= glob.glob('C:/Users/Rena/Desktop/normal/*.jpg')
for myFile in files:
    print(myFile)
    image = cv2.imread(myFile) 
    H,S,V= cv2.split(image)
    Hue= cv2.normalize(H,None,0,255,cv2.NORM_MINMAX)
    X_data.append(Hue)
    
en=enumerate(X_data)

for i , image in en:
    Image.fromarray(image).save(disk_dir / f"{i}.jpg")
    


# Initialising the CNN
model = Sequential() 

#Convolution
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#Pooling
model.add(MaxPooling2D(pool_size = (2, 2))) 

#2nd layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

#3rd layer 
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

#for 6th model
model.add(Dropout(0.3))

#Flattening
model.add(Flatten())#no pars ,it flattens the previous layer

#Full connection
model.add(Dense(units = 128, activation = 'relu')) #unis =output dimension
model.add(Dropout(0.5))
model.add(Dense(units = 1, activation = 'sigmoid')) #final layer,sigmoid->binary outcome
# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#code from keras_documentation

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,  #geometrical transformation
                                   zoom_range = 0.2,
                                   horizontal_flip = True) 

test_datagen = ImageDataGenerator(rescale = 1./255) 

#A DirectoryIterator yielding tuples of (x, y) where x is a numpy array containing a batch of images with shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels.
training_set = train_datagen.flow_from_directory('E:/Πτυχιακη/Step2/Data/H_Channel_endoscopic_data/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary') 

test_set = test_datagen.flow_from_directory('E:/Πτυχιακη/Step2/Data/H_Channel_endoscopic_data/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
model.summary()
model.fit_generator( training_set, 
                     steps_per_epoch = 465,
                     epochs = 4, 
                     validation_data = test_set,
                     validation_steps = 233) 

