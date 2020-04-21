import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
from keras.preprocessing import image
from keras.layers import Dropout
import shap 
from keras import backend as K
import cv2
import glob
from keras.datasets import mnist
import skimage
import io

batch_size = 32
num_classes = 2
epochs = 4

img_rows, img_cols = 64,64
'''
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_dir='C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/Step_1/data/train/abnormal'
test_dir='C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/Step_1/data/test/abnormal/'
train_ab = ['C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/Step_1/data/train/abnormal/*.png'.format(i) for i in os.listdir(train_dir)]
'''


''' it works giving (465,)
DATADIR= "C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/Step_1/data/train"
CATEGORIES= ["abnormal", "normal"]

all_images = []
for category in CATEGORIES:
        path= os.path.join(DATADIR, category)
        for img in os.listdir(path):
          img = cv2.imread(img )
          #img = img.reshape([64, 64, 3])
          all_images.append(img)
        x_train = np.array(all_images)

'''
train = []
train_labels = [] #y_train
files = glob.glob ("C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/Step_1/data/train/abnormal/*.jpg") # your image path
for abnormal in files:
    image = cv2.imread (abnormal)
    train.append (image)
    train_labels.append([1., 0.])
files = glob.glob ("C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/Step_1/data/train/normal/*.jpg")
for normal in files:
    image = cv2.imread (normal)
    train.append (image)
    train_labels.append([0., 1.])
    
train = np.array(train,dtype='float32') #as mnist ,the train is a list i have to make it an array
train = train.reshape(train,[train.shape[0],img_rows*img_cols*3])
print(train.shape)

train_labels = np.array(train_labels,dtype='float64') #as mnist

if K.image_data_format() == 'channels_first':
    train = train.reshape(train.shape[0], 1, img_rows, img_cols)
    input_shape = (1,img_rows, img_cols)
else:
    train = train.reshape(train.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
# convert (number of images x height x width x number of channels) to (number of images x (height * width *3)) 
# for example (120 * 40 * 40 * 3)-> (120 * 4800)
'train = np.reshape(train,[465,64*64*3])

# save numpy array as .npy formats
np.save('train',train)
np.save('train_labels',train_labels)

train = train.astype('float32')
train_labels = train_labels.astype('float32')
train /= 255
train_labels /= 255
print('x_train shape:', train.shape)
print(train.shape[0], 'train samples')
print(test.shape[0], 'test samples')
#Test data
test = []
test_y = []
files = glob.glob ("C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/Step_1/data/test/abnormal/*.png")
for myFile in files:
    image = cv2.imread (myFile)
    test.append (image)
    test_y.append([1., 0.]) # class1
files = glob.glob ("C:/Users/Rena/Desktop/Sxolh/Πτυχιακη/Step_1/data/test/abnormal/*.png")
for myFile in files:
    image = cv2.imread (myFile)
    test.append (image)
    test_y.append([0., 1.]) # class2
          
if K.image_data_format() == 'channels_first':
    test = test.reshape(test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    test = test.reshape(test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
    
  '''  
if K.image_data_format() == 'channels_first':
    train = train.reshape(train.shape[0], 1, img_rows, img_cols)
    test = test.reshape(test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train = train.reshape(train.shape[0], img_rows, img_cols, 1)
    test = test.reshape(test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
 '''   


   





    '''
test = np.array(test,dtype='float32') #as mnist example
test_y = np.array(test_y,dtype='float64') #as mnist
test = np.reshape(test,[233,64*64*3])'''
# convert class vectors to binary class matrices
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_y = keras.utils.to_categorical(test_y, num_classes)






