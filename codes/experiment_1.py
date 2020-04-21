
# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
from keras.preprocessing import image

# Initialising the CNN
model = Sequential() 

#  Convolution & pooling
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2))) 

#2nd layer 
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

'''
for 3rd layer or for 2nd layer
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

'''

# Step 3 - Flattening
model.add(Flatten())#no pars ,it flattens the previous layer

# Step 4 - Full connection
model.add(Dense(units = 128, activation = 'relu')) #128 is up for exp/tion ,units=output dimension
model.add(Dropout(0.5))
model.add(Dense(units = 1, activation = 'sigmoid')) #output layer,sigmoid->binary outcome, else softmax

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#code form keras_documentation

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,  
                                   zoom_range = 0.2,
                                   horizontal_flip = True) 

test_datagen = ImageDataGenerator(rescale = 1./255) 

training_set = train_datagen.flow_from_directory('C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/data/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary') 

test_set = test_datagen.flow_from_directory('C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/data/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

model.fit_generator(training_set, 
                         steps_per_epoch = 465,
                         epochs = 4, 
                         validation_data = test_set,
                         validation_steps = 233) 
'''
    fitting the cnn to the images
'''


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
    
    




