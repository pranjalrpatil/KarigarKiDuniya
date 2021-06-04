# importing libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import keras.preprocessing
#from keras.preprocessing import image
import numpy as np


img_width, img_height = 224, 224
train_data_dir = 'imgdb/train'
validation_data_dir = 'imgdb/test'
nb_train_samples = 35 
nb_validation_samples = 25
epochs = 10
batch_size = 16
  
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
  
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2, 2)))
  
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2, 2)))
  
model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2, 2)))
  
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
  
model.summary()
model.compile(loss ='binary_crossentropy',
                     optimizer ='rmsprop',
                   metrics =['accuracy'])
  
train_datagen = ImageDataGenerator(
                rescale = 1. / 255,
                 shear_range = 0.2,
                  zoom_range = 0.2,
            horizontal_flip = True)
  
test_datagen = ImageDataGenerator(rescale = 1. / 255)
  
train_generator = train_datagen.flow_from_directory(train_data_dir,
                              target_size =(img_width, img_height),
                     batch_size = batch_size, class_mode ='binary')
  
validation_generator = test_datagen.flow_from_directory(
                                    validation_data_dir,
                   target_size =(img_width, img_height),
          batch_size = batch_size, class_mode ='binary')
  
model.fit_generator(train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs, validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)




# Saving the model for Future Inferences

model_json = model.to_json()
with open("model_saved.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('model_saved.h5')


##############################################
# imports

from keras.models import model_from_json 

# opening and store file in a variable

json_file = open('model_saved.json','r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model_saved.h5")
print("Loaded Model from disk")

# compile and evaluate loaded model

loaded_model.compile(loss ='binary_crossentropy',optimizer ='rmsprop',metrics =['accuracy'])
