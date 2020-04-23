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
training_set = train_datagen.flow_from_directory('E:/Πτυχιακη/Step_1/HSV_endoscopic_data/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary') 

test_set = test_datagen.flow_from_directory('E:/Πτυχιακη/Step_1/HSV_endoscopic_data/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
model.summary()
model.fit_generator( training_set, 
                     steps_per_epoch = 465,
                     epochs = 4, 
                     validation_data = test_set,
                     validation_steps = 233) 
# testing 
import os
import cv2
from pathlib import Path 
from PIL import Image
import csv
import glob

#renaming images , needs to be done for every subfolder 
    i = 0
      
    for filename in os.listdir('E:/Πτυχιακη/Step_1/HSV_endoscopic_data/test/normal/'): 
        dst ="endo" + str(i) + ".jpg"
        
        #src ='C:/Users/Rena/Desktop/bla/'+ filename 
        #dst ='C:/Users/Rena/Desktop/bla/'+ dst 
        src ='E:/Πτυχιακη/Step_1/HSV_endoscopic_data/test/normal/'+ filename 
        dst ='E:/Πτυχιακη/Step_1/HSV_endoscopic_data/test/normal/'+ dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
        
        
#where to save the images

#disk_dir= Path('C:/Users/Rena/Desktop/bla/converted/')
disk_dir= Path('E:/Πτυχιακη/Step_1/HSV_endoscopic_data/train/normal/')
'''
img = cv2.imread('C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/Step_1/data/test/abnormal/im.jpg')
cv2.imshow('original image',HSV_img)
cv2.waitKey(0) #these to lines prevent it from crashing
cv2.destroyAllWindows()

HSV_img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
Image.fromarray(HSV_img).save(disk_dir / f"{7}.png")


for i , image in enumerate(training_set):
    Image.fromarray(image).save(disk_dir / f"{i}.png")
 '''
#read from that path + convert
X_data = [] 
#C:\Users\Rena\Desktop\normal
#E:\Πτυχιακη\Step_1\data\train\abnormal
files= glob.glob('C:/Users/Rena/Desktop/normal/*.jpg')
for myFile in files:
    print(myFile)
    image = cv2.imread (myFile) 
    HSV_img = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    X_data.append(HSV_img)
    
#print('X_data shape:', np.array(training_set).shape)

en=enumerate(X_data)
#print ("Return type:",type(en)) 
#print (list(enumerate(en)))

#saving the images
for i , image in en:
    Image.fromarray(image).save(disk_dir / f"{i}.jpg")
