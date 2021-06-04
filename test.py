import pandas as pd
import numpy as np 
import itertools
import tensorflow as tf
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense 
from keras import applications 
from keras.utils.np_utils import to_categorical 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math 
import datetime
import time

########################################################################################
image_path = 'flask/v_data/test/shawls/311.jpg'

orig = mpimg.imread(image_path)

print("[INFO] Image Loaded")
image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image)

# important! otherwise the predictions will be '0'
image = image / 255

image = np.expand_dims(image, axis=0)
# print(image)
# build the VGG16 network
model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')

datagen = ImageDataGenerator(rescale=1. / 255)
#Default dimensions we found online
img_width, img_height = 224, 224 
# batch size used by flow_from_directory and predict_generator 
batch_size = 16

# get the bottleneck prediction from the pre-trained VGG16 model
bottleneck_prediction = model.predict(image)
train_data_dir = 'flask/v_data/train' 

generator_top = datagen.flow_from_directory( 
   train_data_dir, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=False) 

num_classes = len(generator_top.class_indices) 

top_model_weights_path = 'bottlenecksss_fc_model.h5'

# build top model  
model = Sequential()  
model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))  
model.add(Dense(100, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(50, activation='relu'))  
model.add(Dense(num_classes, activation='softmax'))


model.load_weights(top_model_weights_path)

# use the bottleneck prediction on the top model to get the final classification  
class_predicted = model.predict_classes(bottleneck_prediction)

inID = class_predicted[0]  

class_dictionary = generator_top.class_indices  

inv_map = {v: k for k, v in class_dictionary.items()}  

label = inv_map[inID]  

# get the prediction label  
print("Image ID: {}, Label: {}".format(inID, label))