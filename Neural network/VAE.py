#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 12:07:38 2020

@author: chenghaili
"""

import os
os.chdir(os.path.abspath(os.path.join(os.getcwd(), "..")))

import args
import time
import numpy as np
import tensorflow as tf
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

train_size = 800
BATCH_SIZE = 32
BUFFER_SIZE = 200

train_data = data[:train_size]
test_data = data[train_size:]

train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(BATCH_SIZE)

class CVAE(tf.keras.Model):
    
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(128, 128, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=5, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ])

        self.generative_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=16*16*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(16, 16, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=5, strides=(2, 2), padding="SAME", activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=5, strides=(2, 2), padding="SAME", activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=5, strides=(2, 2), padding="SAME", activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
        ])


    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z):
        logits = self.generative_net(z)
        return logits

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

@tf.function
def compute_loss(model, x):
    
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    #cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    cross_ent = tf.keras.losses.MSE(x_logit, x)   
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  

if __name__=='__main__':
    
    latent_dim = 100
    model = CVAE(latent_dim)
    
    EPOCH = 500
    optimizer = tf.keras.optimizers.Adam(1e-3)
    elbo_list = []
    
    for epoch in range(1, EPOCH + 1):
        start_time = time.time()
        
        for train_x in train_dataset:
            compute_apply_gradients(model, train_x, optimizer)
            
        end_time = time.time()
    
        if epoch % 1 == 0:
            
            loss = tf.keras.metrics.Mean()
            for train_x in train_dataset:
                loss(compute_loss(model, train_x))
    
            elbo = -loss.result()
            elbo_list.append(elbo)
            print('Epochs {} ELBO: {:.4f}'.format(epoch, elbo.numpy()))
            
    model.save_weights(path+'VAE_ckpt')

# gener = model.generative_net(tf.random.normal(shape=[1, latent_dim]))
# plt.imshow(gener[0, :, :, 0])

# mean, logvar = model.encode(tf.expand_dims(train_data[0], 0))
        
# mean, logvar = model.encode(test_data)
# plt.hist(mean.numpy().flatten(), bins=200, color='steelblue')
# plt.savefig(path+'mean.png', dpi = 300)
        
# plt.hist(np.exp(logvar.numpy()).flatten(), bins=200, color='steelblue')
# plt.savefig(path+'var.png', dpi = 300)

# z = model.reparameterize(mean, logvar)
# x_logit = model.decode(z)
# loss = tf.keras.losses.MAE(x_logit, tf.expand_dims(train_data[0], 0)) 
