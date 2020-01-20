#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 01:14:20 2020

@author: chenghaili
"""

import os
os.chdir(os.path.abspath(os.path.join(os.getcwd(), "..")))

import args
import numpy as np
from matplotlib import  pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

n = args.n
size = args.size
path = args.path

BATCH_SIZE = 32
BUFFER_SIZE = 200
train_size = 800
comp = 100

data = np.load(path+'data.npy')
image = []
for porous in data:
    image.append(porous.flatten())
    
X_std = StandardScaler().fit_transform(image)
pca = PCA(n_components = comp)
principalComponents = pca.fit_transform(image)

rho_diff = np.load(path+'rho_diff.npy')
label = rho_diff.T[:, -1]
label = tf.convert_to_tensor(label)

train_data = principalComponents[:train_size]
test_data = principalComponents[train_size:]

train_label = label[:train_size]
test_label = label[train_size:]

def get_model():
    
    model = tf.keras.Sequential()

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(1, activation=None))

    return model

model = get_model()

loss_function = tf.keras.losses.MAE   
optimizer = tf.keras.optimizers.Adam(1e-3)
EPOCHS = 500

model.compile(loss=loss_function, optimizer=optimizer)
history1 = model.fit(train_data, train_label,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          shuffle=True,
          validation_data=(test_data, test_label))

optimizer = tf.keras.optimizers.Adam(2e-4)

model.compile(loss=loss_function, optimizer=optimizer)
history2 = model.fit(train_data, train_label,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          shuffle=True,
          validation_data=(test_data, test_label))

loss = np.array(history1.history['loss'] + history2.history['loss'])
val_loss = np.array(history1.history['val_loss'] + history2.history['val_loss'])

# plt.plot(np.log(loss))
# plt.plot(np.log(val_loss))

plt.figure(figsize=(10, 10))
for i in range (200):
    plt.scatter(model.predict(tf.expand_dims(test_data[i], 0))[0][0], test_label[i].numpy())
plt.plot([1, 3], [1, 3])
plt.savefig(path+'result.png', dpi=300)