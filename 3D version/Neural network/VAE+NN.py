#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 13:38:26 2020

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

EPOCHS = 300
BATCH_SIZE = 32
BUFFER_SIZE = 200

train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(BATCH_SIZE)

from VAE import CVAE

VAE_model = CVAE(100)
VAE_model.load_weights(path + 'checkpoints/VAE/VAE_ckpt')
enc = VAE_model.inference_net


inputs = tf.keras.Input(shape=(128,128, 1), name='img_input')

x = enc(inputs)[:, :100]
x = layers.Dense(512, activation='relu')(x)
#x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
#x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
#x = layers.Dropout(0.3)(x)
x = layers.Dense(1)(x)

model = tf.keras.Model(inputs=inputs,outputs=x)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.MAE  
    )

history1 = model.fit(train_data, train_label,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          shuffle=True,
          validation_data=(test_data, test_label))

model.optimizer.lr = 2e-4

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

