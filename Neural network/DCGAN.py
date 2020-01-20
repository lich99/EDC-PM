#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:41:59 2020

@author: chenghaili
"""

import os
os.chdir(os.path.abspath(os.path.join(os.getcwd(), "..")))

import args
import time
import numpy as np
from matplotlib import  pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

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

noise_dim = 100
BATCH_SIZE = 10
BUFFER_SIZE = 50
EPOCHS = 200

train_dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator_model():
    
    model = tf.keras.Sequential()
    
    model.add(layers.Dense( 16 * 16 * 64, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 64)))

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (10, 10), strides=(4, 4), padding='same', use_bias=False, activation='sigmoid'))

    return model

def make_discriminator_model():
    
    model = tf.keras.Sequential()
    
    model.add(layers.Conv2D(32, (10, 10), strides=(4, 4), padding='same', input_shape=[128, 128, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

generator = make_generator_model()
discriminator = make_discriminator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return tf.reduce_sum(gen_loss), tf.reduce_sum(disc_loss)
    
def train(dataset, epochs):
    
    for epoch in range(epochs):
        start = time.time()
        g = 0
        d = 0
        for image_batch in dataset:
            gen_l, disc_l = train_step(image_batch)
            g += gen_l
            d += disc_l
        print ('epoch {} use {:.5f} sec, G_l {:.5f}, D_l {:.5f}'.format(epoch + 1, time.time()-start, g, d))
    
generator_optimizer = tf.keras.optimizers.Adam(1e-3)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-5)
       
train(train_dataset, EPOCHS)

noise = tf.random.normal([1, noise_dim])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0])
decision = discriminator(generated_image)
print (decision)