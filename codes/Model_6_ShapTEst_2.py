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

training_set = train_datagen.flow_from_directory('C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/Step_1/data/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary') 

test_set = test_datagen.flow_from_directory('C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/Step_1/data/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
model.summary()
model.fit_generator( training_set, 
                     steps_per_epoch = 465,
                     epochs = 4, 
                     validation_data = test_set,
                     validation_steps = 233) 
#shap 
model.fit(x=None,
    y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1)

model.fit(train_datagen,training_set,
          batch_size=32,
          epochs=4,
          verbose=1,
          validation_data=(test_datagen,test_set))

import shap 

# select a set of background examples to take an expectation over
background = training_set[np.random.choice(training_set[0], 100, replace=False)]

# explain predictions of the model on three images
e = shap.DeepExplainer(model, background)
# ...or pass tensors directly
# e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
shap_values = e.shap_values(test_set[1:5])

# plot the feature attributions
shap.image_plot(shap_values, -test_set[1:5])


#try shit_2 kerne l expl

kernel_explainer = shap.KernelExplainer(model.predict, training_set[:10])
kernel_shap_values = kernel_explainer.shap_values(test_set[:1])

x_test_words = prepare_explanation_words(model,test_set)
y_pred = model.predict(test_set[:1])
print('Actual Category: %s, Predict Category: %s' % (y_test[0], y_pred[0]))

shap.force_plot(kernel_explainer.expected_value[0], kernel_shap_values[0][0], x_test_words[0])

#try shit DEEP SHAP

explainer = shap.DeepExplainer(model, training_set[:10])

