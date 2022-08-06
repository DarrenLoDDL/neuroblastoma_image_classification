import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from numpy import *
import os
from PIL import Image 

from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import np_utils, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import ResNet50

print('ok')

train_directory ='F:/Datasets/nbt_split/train' #dataset locations may vary
val_directory='F:/Datasets/nbt_split/validation'
test_directory='F:/Datasets/nbt_split/test'


##load in training images
train_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_directory,
    target_size=(150,150),
    batch_size=32)

##load in validation
val_datagen = ImageDataGenerator(rescale=1./255)

val_gen = val_datagen.flow_from_directory(
    val_directory, 
    target_size=(150,150),
    batch_size=32
    
)

##load in test
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    test_directory,
    target_size=(150,150),
    batch_size=32
)

base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(150,150,3),
    pooling='max',
    classes = 5,
   
)

base_model.trainable = False

new_model = keras.Sequential()
new_model.add(base_model)
new_model.add(keras.layers.Flatten())
new_model.add(keras.layers.Dropout(0.25))
new_model.add(keras.layers.Dense(16))
new_model.add(keras.layers.Dense(5, activation="softmax"))
    
new_model.summary()

opt = keras.optimizers.SGD(learning_rate =0.005 , momentum = 0.3)

new_model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics =['accuracy'],
)

history = new_model.fit_generator(train_gen, steps_per_epoch=100, epochs=55,
                              validation_data=val_gen, validation_steps=50,  verbose =1)    


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='ACC (testing data)')
plt.plot(history.history['val_accuracy'], label='ACC (validation data)')
plt.title('Training and Validation Loss/Accuracy')
plt.ylabel('ACC')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

new_model.save('trained_fc.h5')


