#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:45:22 2020

@author: chenghaili
"""

import os
os.chdir(os.path.abspath(os.path.join(os.getcwd(), "..")))

import args
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import  pyplot as plt

n = args.n
size = args.size
path = args.path

BATCH_SIZE = 32
BUFFER_SIZE = 200
EPOCHS = 300
train_size = 800
comp = 250

data = np.load(path+'data.npy')
image = []
for porous in data:
    image.append(porous.flatten())
    
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(image)
pca = PCA(n_components=comp)
principalComponents = pca.fit_transform(image)
principalComponents = tf.convert_to_tensor(principalComponents)

train_pca = principalComponents[:train_size]
test_pca = principalComponents[train_size:]

data = tf.convert_to_tensor(data)
data = tf.expand_dims(data, -1)

train_data = data[:train_size]
test_data = data[train_size:]

rho_diff = np.load(path+'rho_diff.npy')
label = rho_diff.T[:, -1]
label = tf.convert_to_tensor(label)

train_label = label[:train_size]
test_label = label[train_size:]

image_input = tf.keras.Input(shape=(128,128, 1), name='img_input')
pca_input = tf.keras.Input(shape=(comp), name='pca_input')

x = layers.Conv2D(64, (1, 1), activation='relu', padding='same', name='cov1')(image_input)
x = layers.Conv2D(64, (5, 5), activation='relu', padding='same', name='cov2')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((4, 4))(x)

x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)
y = tf.concat([x, pca_input], 1)
y = layers.Dense(512, activation='relu')(y)
y = layers.Dropout(0.2)(y)
y = layers.Dense(1, activation=None)(y)

model = tf.keras.Model(inputs=[image_input, pca_input],outputs=y)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.MAE  
    )

history1 = model.fit([train_data,train_pca], train_label,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          shuffle=True,
          validation_data=([test_data, test_pca], test_label))

model.optimizer.lr = 2e-4

history2 = model.fit([train_data,train_pca], train_label,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          shuffle=True,
          validation_data=([test_data, test_pca], test_label))

loss = np.array(history1.history['loss'] + history2.history['loss'])
val_loss = np.array(history1.history['val_loss'] + history2.history['val_loss'])

# plt.plot(np.log(loss))
# plt.plot(np.log(val_loss))

plt.figure(figsize=(10, 10))
for i in range (200):
    plt.scatter(model.predict([tf.expand_dims(test_data[i], 0), tf.expand_dims(test_pca[i], 0)])[0][0], test_label[i].numpy())
plt.plot([1, 3], [1, 3])
plt.savefig(path+'result.png', dpi=300)


# class My_model(tf.keras.Model):

#   def __init__(self, training = True):
#     super(My_model, self).__init__(name='My_model')
    
#     self.cov1 = layers.Conv2D(64, (1, 1), activation='relu', padding='same', input_shape=(128, 128, 1))
#     self.cov2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')

#     self.cov3 = layers.Conv2D(128, (1, 1), activation='relu', padding='same')
#     self.cov4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')

#     self.cov5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
#     self.cov6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')

#     self.fc1 = layers.Dense(512, activation='relu')
#     self.fc2 = layers.Dense(512, activation='relu')
#     self.fc3 = layers.Dense(1, activation=None)
    
#   def call(self, inputs):
      
#     image, pca = inputs
      
#     x = self.cov1(image)
#     x = self.cov2(x)
#     x = layers.MaxPooling2D((2, 2))(x)
    
#     x = self.cov3(x)
#     x = self.cov4(x)
#     x = layers.MaxPooling2D((2, 2))(x)
    
#     x = self.cov5(x)
#     x = self.cov6(x)
#     x = layers.MaxPooling2D((4, 4))(x)
    
#     x = layers.Flatten()(x)
#     x = self.fc1(x)
#     x = layers.Dropout(0.2)(x)
#     y = tf.concat([x, pca], 1)
#     y = self.fc2(y)
#     y = layers.Dropout(0.2)(y)
#     y = self.fc3(y)
    
#     return y