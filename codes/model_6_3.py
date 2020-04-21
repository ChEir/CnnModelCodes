"""
Created on Sun Feb 10 16:01:36 2019

@author: Rena
"""

# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
from keras.preprocessing import image
from keras.layers import Dropout

# Initialising the CNN
model = Sequential() 

#Convolution
model.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))

#Pooling
model.add(MaxPooling2D(pool_size = (2, 2))) 

#2nd layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

#3rd layer 
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

#4th layer
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
#5th layer
model.add(Conv2D(128, (3, 3), activation = 'relu'))
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

training_set = train_datagen.flow_from_directory('C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/Step_1/data/train',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary') 

test_set = test_datagen.flow_from_directory('C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/Step_1/data/test',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')
model.summary()
model.fit_generator( training_set, 
                     steps_per_epoch = 465,
                     epochs = 4, 
                     validation_data = test_set,
                     validation_steps = 233) 