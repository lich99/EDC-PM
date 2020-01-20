#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:03:38 2020

@author: chenghaili
"""

import os
os.chdir(os.path.abspath(os.path.join(os.getcwd(), "..")))

import args
import numpy as np
import tensorflow as tf
from matplotlib import  pyplot as plt
from tensorflow.keras import layers

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config) 

# Init args
n = args.n
size = args.size
drug = args.drug
drug_quantity = args.drug_quantity
porosity = args.porosity
blobiness = args.blobiness
path = args.path

data = np.load(path+'data.npy')
data = tf.convert_to_tensor(data)
data = tf.expand_dims(data, -1)

rho_left = np.load(path+'rho_left.npy')
rho_accu = np.load(path+'rho_accu.npy')
rho_diff = np.load(path+'rho_diff.npy')

label = rho_diff.T[:, -1]
label = tf.convert_to_tensor(label)

train_size = 800

train_data = data[:train_size]
test_data = data[train_size:]

train_label = label[:train_size]
test_label = label[train_size:]

BATCH_SIZE = 32
BUFFER_SIZE = 200
EPOCHS = 200

def get_model():
    
    model = tf.keras.Sequential()
    
    model.add(layers.Conv2D(64, (1, 1), activation='relu', padding='same', input_shape=(128, 128, 1)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (1, 1), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((4, 4)))

    model.add(layers.Flatten())  
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation=None))

    return model

model = get_model()

loss_function = tf.keras.losses.MAE   
optimizer = tf.keras.optimizers.Adam(1e-3)

model.compile(loss=loss_function, optimizer=optimizer)
model.fit(train_data, train_label,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(test_data, test_label))

optimizer = tf.keras.optimizers.Adam(1e-4)

model.compile(loss=loss_function, optimizer=optimizer)
model.fit(train_data, train_label,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(test_data, test_label))

for i in range (200):
    plt.scatter(model.predict(tf.expand_dims(test_data[i], 0))[0][0], test_label[i].numpy())
plt.plot([1, 3], [1, 3])