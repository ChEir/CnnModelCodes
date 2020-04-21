# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense #to add F-con layer

# Initialising the CNN
classifier = Sequential() 

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))#in_sh the format in which the im will be converted,3=rgb

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2))) #reduce the size of maps, div by 2

# Adding a second convolutional layer ,applied on the maps of the previous steps
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())#no pars ,it flattens the previous layer

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu')) #128 is up for exp/tion ,units=output dimension
classifier.add(Dense(units = 1, activation = 'sigmoid')) #output layer,sigmoid->binary outcome, else softmax

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) #adam-gradient desc,here the model is ready to be compiled

# Part 2 - Fitting the CNN to the images,im-preproc
#use keras documentation to im augmentation, preproc-im to prevent overfitting
#ImageDataGenerator class ,https://keras.io/preprocessing/image/ 
#creates batches of my ims and adds random transformations (rotate,flip)
#code below copied from .flow_from_directory(directory)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,#geometrical transf
                                   zoom_range = 0.2,
                                   horizontal_flip = True) #also vertical flip 
#when you run this ,an object is created that id gonna be used to augment the ims of the training set


test_datagen = ImageDataGenerator(rescale = 1./255) #apply augmentation also on train set 
#this object is used to preproc the im of the test set 

training_set = train_datagen.flow_from_directory('E:/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary') 
#where we extract the im from,size of im expected in CNN model,size of batches in which some random samples will be included,
#when run,we apply the augmentation,resizing etc
#attention on the path,possible replacement of \ with /


test_set = test_datagen.flow_from_directory('E:/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

#the fit_gen is applied on our CNN model/object ="classifier"
classifier.fit_generator(training_set, 
                         steps_per_epoch = 465,#num of im in train set
                         epochs = 2, #25 after an eternity
                         validation_data = test_set,
                         validation_steps = 233) #num of im in test set

#fit the CNN to our training set + testing its performance on the test set

#look at the acc of the test set, and the diff between the acc of test & training to see if there is overfitting

#goal : increase the acc of test > 80% and decrease the diff between the 2 accs

#ways to improve : add another conv layer +(apply pooling too )or another f-con layer 
#in the added conv layer you can double the num of filters, 32,64(on 3rd layer)...

#another way is to change at line 63/72 the target size,to bigger , to get more info of the pixel patterns

import numpy as np #we are using a function that is gonna preprocess the imgae we are gonna load 
from keras.preprocessing import image
test_im = image.load_img('path', target_size=(64,64)) #same dims we used in training 
#add a new dim cause of rgb ,has the same format 

test_im = image.img_to_array(test_im)
 #aadd one more dim , to be able to use the predict method ,to give what the predict method expects , it corresponds to the num of the batch 
 test_im = np.expand_dims(test_im, axis =0 )
 result= classifier.predict(test_im)
 #what the res show?
 
 training_Set.class_indices #shows what corresponds to what 
 
 if result[0][0] == 1:
     prediction ='normal'
else:
    prediction = 'abnormal'
    
    




