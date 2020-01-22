#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 19:04:22 2020

@author: chenghaili
"""

import os
os.chdir(os.path.abspath(os.path.join(os.getcwd(), "..")))  

import args
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True) 

dtype = tf.float32

# Init LBM

lx = args.size[0]           
ly = args.size[1]        
lz = args.size[2]     
tau = 1                      

e_i = np.zeros((19, 3), dtype = np.int32)
opp = np.zeros((19), dtype = np.int32)

e_i[0]  = [ 0, 0, 0];      opp[0]  = 0
e_i[1]  = [ 1, 0, 0];      opp[1]  = 2
e_i[2]  = [-1, 0, 0];      opp[2]  = 1
e_i[3]  = [ 0, 0, 1];      opp[3]  = 4
e_i[4]  = [ 0, 0,-1];      opp[4]  = 3
e_i[5]  = [ 0,-1, 0];      opp[5]  = 6
e_i[6]  = [ 0, 1, 0];      opp[6]  = 5
e_i[7]  = [ 1, 0, 1];      opp[7]  = 10
e_i[8]  = [-1, 0, 1];      opp[8]  = 9
e_i[9]  = [ 1, 0,-1];      opp[9]  = 8
e_i[10] = [-1, 0,-1];      opp[10] = 7
e_i[11] = [ 1,-1, 0];      opp[11] = 14
e_i[12] = [-1,-1, 0];      opp[12] = 13
e_i[13] = [ 1, 1, 0];      opp[13] = 12
e_i[14] = [-1, 1, 0];      opp[14] = 11
e_i[15] = [ 0,-1, 1];      opp[15] = 18
e_i[16] = [ 0,-1,-1];      opp[16] = 17
e_i[17] = [ 0, 1, 1];      opp[17] = 16
e_i[18] = [ 0, 1,-1];      opp[18] = 15

w_i = np.array([1/3, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
                1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, ])

e_i = tf.convert_to_tensor(e_i, dtype = tf.int32) 
opp = tf.convert_to_tensor(opp, dtype = tf.int32) 
w_i = tf.convert_to_tensor(w_i, dtype = dtype)   

path = args.path

def equilibrium(rho):
    
    return tf.stack([rho * w_i[i]  for i in range (19)], axis = 4)

def set_zero_boundary(num, max_x, max_y, max_z):
    
    b = np.ones((num, max_x, max_y, max_z, 19))
    b[:, -1, :, :, :] = 0
    return tf.convert_to_tensor(b, dtype = dtype)

def Init_velo(num, max_x, max_y, max_z):
    
    uy = np.zeros((num, max_x, max_y, max_z))
    ux = np.zeros((num, max_x, max_y, max_z))
    uz = np.zeros((num, max_x, max_y, max_z))
    
    return tf.convert_to_tensor(ux, dtype = dtype), tf.convert_to_tensor(uy, dtype = dtype), \
           tf.convert_to_tensor(uz, dtype = dtype)

def Init_rho(drug):

    rho = drug
    return rho

def run(Iter, start, end):
    
    data = np.load(path+'data.npy')[start:end]
    n = data.shape[0]
    
    filled_temp = np.zeros_like(data)
    filled_temp[data == 1] = 1

    filled = np.zeros((n, lx+2, ly+2, lz+2))
    
    filled[:, 1:-1, 1:-1, 1:-1] = filled_temp
    filled[:, 1:, 0, :] = 1
    filled[:, 1:, -1, :] = 1
    filled[:, 1:, :, 0] = 1
    filled[:, 1:, :, -1] = 1
    
    drug = np.zeros((n, lx+2, ly+2, lz+2))
    drug[:, 0, :, :] = 1

    spacing = tf.convert_to_tensor(filled == 0, dtype = dtype)
    filled = tf.convert_to_tensor(filled, dtype = dtype)
    filled = tf.expand_dims(filled, 4)
    drug = tf.convert_to_tensor(drug, dtype = dtype)

    zero_boundary = set_zero_boundary(n, lx+2, ly+2, lz+2)
    rho = Init_rho(drug)
    # ux, uy, uz = Init_velo(n, lx+2, ly+2, lz+2)
    
    # cal_accu = np.zeros((n, lx+2, ly+2, lz+2, 19))
    # cal_accu[:, :, -1, 2] = 1
    # cal_accu[:, :, -1, 5] = 1
    # cal_accu[:, :, -1, 6] = 1
    # cal_accu = tf.convert_to_tensor(cal_accu, dtype = dtype)
    
    rho = rho * spacing
    f_i = equilibrium(rho)
    f_eq = f_i
    
    # ux = ux * spacing
    # uy = uy * spacing
    # uz = uz * spacing
    
    spacing = tf.expand_dims(spacing, 4)
    
    rho_l = []
    # rho_a = []
    rho_l.append(tf.reduce_sum(rho, axis = [1, 2, 3]))
    # rho_a.append(tf.reduce_sum(rho, axis = [1, 2]))

    # accu = 0
    start = time.time()

    for t in range(Iter):

        f_out = f_i - (f_i - f_eq) / tau
        f_i = tf.stack([tf.roll(tf.roll(tf.roll(f_out[:, :, :, :, i], e_i[i, 0], axis=1), e_i[i, 1], axis=2), e_i[i, 2], axis=3) for i in range (19)], axis=4)

        # accu += tf.reduce_sum(f_i * cal_accu, axis = [1, 2, 3])
        f_i = f_i * zero_boundary

        temp1 = f_i * filled
        temp2 = tf.stack([tf.roll(tf.roll(tf.roll(temp1[:, :, :, :, opp[i]], e_i[i, 0], axis=1), e_i[i, 1], axis=2), e_i[i, 2], axis=3) for i in range (19)], axis=4)
        f_i = f_i * spacing + temp2
        rho = tf.reduce_sum(f_i, axis=4)
        rho = tf.maximum(rho, drug)

        now = tf.reduce_sum(rho, axis = [1, 2, 3])
        rho_l.append(now.numpy())
        # rho_a.append((now + accu).numpy())

        f_eq = equilibrium(rho)

        if t % 10 == 9:
            print('Iter: {} Runtime: {:.5f} sec'.format(t+1, time.time() - start))
            start = time.time() 

    # return rho.numpy(), rho_l, rho_a, filled
    return rho.numpy(), rho_l, filled


if __name__ == '__main__':
    
    Iter = 250
    
    a = time.time()    
    with tf.device('/device:gpu:0'):
        # rho, rho_left, rho_accu, _ = run(Iter, 0, 2)
        rho, rho_left, _ = run(Iter, 0, args.n)
    b = time.time()
    
    print('Total Runtime: {} sec'.format(b-a))
    
    # rho_diff = []
    # for i in range (Iter):
    #     rho_diff.append(rho_accu[i+1] - rho_accu[i])
    
    rho_left = np.array(rho_left)
    # rho_accu = np.array(rho_accu)
    # rho_diff = np.array(rho_diff)
    
    np.save(path+'rho.npy', rho)
    np.save(path+'rho_left.npy', rho_left)
    # np.save(path+'rho_accu.npy', rho_accu)
    # np.save(path+'rho_diff.npy', rho_diff)
    
    print('Saved !')


