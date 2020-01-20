#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 19:59:52 2020

@author: chenghaili
"""



import args
import time
import numpy as np
from matplotlib import  pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, regularizers

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
data[data == 1] = -1
data[data == 2] = 1
data = tf.convert_to_tensor(data)
data = tf.expand_dims(data, -1)

rho_left = np.load(path+'rho_left.npy')
rho_accu = np.load(path+'rho_accu.npy')
rho_diff = np.load(path+'rho_diff.npy')

label = rho_diff.T[:, -1]
label = tf.convert_to_tensor(label)

train_data = data[:800]
test_data = data[800:]

train_label = label[:800]
test_label = label[800:]

BATCH_SIZE = 50
BUFFER_SIZE = 500
EPOCHS = 200

# train_dataset = tf.data.Dataset.from_tensor_slices(train_data, train_label).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()

        self.cov0 = layers.Conv2D(64, (1,1), activation='relu', padding='same')
        
        self.cov1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.cov2 = layers.Conv2D(64, (3,3), activation=None, padding='same')
        self.bn2 = layers.BatchNormalization()
        
        self.cov3 = layers.Conv2D(64, (3,3), activation='relu', padding='same')
        self.bn3 = layers.BatchNormalization()
        self.cov4 = layers.Conv2D(64, (3,3), activation=None, padding='same')
        self.bn4 = layers.BatchNormalization()
        
        self.cov5 = layers.Conv2D(64, (3,3), activation='relu', padding='same')
        self.bn5 = layers.BatchNormalization()
        self.cov6 = layers.Conv2D(64, (3,3), activation=None, padding='same')
        self.bn6 = layers.BatchNormalization()
        
        self.fc1 = layers.Dense(1024, activation=tf.keras.activations.relu)
        self.fc2 = layers.Dense(1)
   
    def call(self, inputs):
        
        x = self.cov0(inputs)
        
        y = self.cov1(x)
        y = self.bn1(y)
        y = self.cov2(y)
        y = self.bn2(y)
        x = layers.Add()([x, y])
        x = layers.Activation('relu')(x)
        x = tf.keras.layers.AvgPool2D((2,2))(x)
        
        y = self.cov3(x)
        y = self.bn3(y)
        y = self.cov4(y)
        y = self.bn4(y)
        x = layers.Add()([x, y])
        x = layers.Activation('relu')(x)
        x = tf.keras.layers.AvgPool2D((2,2))(x)
        
        y = self.cov5(x)
        y = self.bn5(y)
        y = self.cov6(y)
        y = self.bn6(y)
        x = layers.Add()([x, y])
        x = layers.Activation('relu')(x)
        x = tf.keras.layers.AvgPool2D((2,2))(x)
        
        

        x = layers.Flatten()(x)
        x = self.fc1(x)
        x = layers.Dropout(0.4)(x)
        x = self.fc2(x)
        
        return x

model = ResNet()

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
    plt.scatter(model.predict(tf.expand_dims(test_data[i], 0))[0][0], test_label[i])
plt.plot([1, 3], [1, 3])


