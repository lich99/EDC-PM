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

dtype = tf.float32

# Init LBM

lx = args.size[0]              # Lattice points in x-direction
ly = args.size[1]              # Lattice points in y-direction
tau = 1                        # relaxation parameter

e_i = np.zeros((9, 2), dtype = np.int32)
opp = np.zeros((9), dtype = np.int32)

e_i[0] = [ 0,  0];      opp[0] = 0
e_i[1] = [ 1,  0];      opp[1] = 3
e_i[2] = [ 0,  1];      opp[2] = 4
e_i[3] = [-1,  0];      opp[3] = 1
e_i[4] = [ 0, -1];      opp[4] = 2
e_i[5] = [ 1,  1];      opp[5] = 7
e_i[6] = [-1,  1];      opp[6] = 8
e_i[7] = [-1, -1];      opp[7] = 5
e_i[8] = [ 1, -1];      opp[8] = 6

w_i = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

e_i = tf.convert_to_tensor(e_i, dtype = tf.int32) 
opp = tf.convert_to_tensor(opp, dtype = tf.int32) 
w_i = tf.convert_to_tensor(w_i, dtype = dtype)   

path = args.path

def equilibrium(rho):
    
    return tf.stack([rho * w_i[i]  for i in range (9)], axis = 3)

def set_zero_boundary(num, max_x, max_y):
    
    b = np.ones((num, max_x, max_y, 9))
    b[:, :, -1, :] = 0
    return tf.convert_to_tensor(b, dtype = dtype)

def Init_velo(num, max_x, max_y):
    
    uy = np.zeros((num, max_x, max_y))
    ux = np.zeros((num, max_x, max_y))
    return tf.convert_to_tensor(ux, dtype = dtype), tf.convert_to_tensor(uy, dtype = dtype)

def Init_rho(drug):

    rho = drug
    return rho

def run(Iter, start, end):
    
    data = np.load(path+'data.npy')[start:end]
    n = data.shape[0]
    
    filled_temp = np.zeros_like(data)
    filled_temp[data == 1] = 1

    filled = np.zeros((n, lx+2, ly+2))
    filled[:, 1:-1, 1:-1] = filled_temp
    filled[:, 0, :] = 1
    filled[:, -1, :] = 1
    drug = np.zeros((n, lx+2, ly+2))
    drug[:, :, 0] = 1

    spacing = tf.convert_to_tensor(filled == 0, dtype = dtype)
    filled = tf.convert_to_tensor(filled, dtype = dtype)
    drug = tf.convert_to_tensor(drug, dtype = dtype)

    zero_boundary = set_zero_boundary(n, lx+2, ly+2)
    rho = Init_rho(drug)
    ux, uy = Init_velo(n, lx+2, ly+2)

    cal_accu = np.zeros((n, lx+2, ly+2, 9))
    cal_accu[:, :, -1, 2] = 1
    cal_accu[:, :, -1, 5] = 1
    cal_accu[:, :, -1, 6] = 1
    cal_accu = tf.convert_to_tensor(cal_accu, dtype = dtype)

    f_i = equilibrium(rho)
    f_eq = f_i
    ux = ux * spacing 
    uy = uy * spacing 
    rho = rho * spacing

    rho_l = []
    rho_a = []
    rho_l.append(tf.reduce_sum(rho, axis = [1, 2]))
    rho_a.append(tf.reduce_sum(rho, axis = [1, 2]))

    accu = 0
    start = time.time() 

    for t in range(Iter):

        f_out = f_i - (f_i - f_eq) / tau
        f_i = tf.stack( [ tf.roll( tf.roll( f_out[:, :, :, i], e_i[i, 0], axis=1), e_i[i, 1], axis=2) for i in range (9)], axis=3)

        accu += tf.reduce_sum(f_i * cal_accu, axis = [1, 2, 3])
        f_i = f_i * zero_boundary

        temp1 = f_i * tf.expand_dims(filled, 3)
        temp2 = tf.stack( [ tf.roll( tf.roll( temp1[:, :, :, opp[i]], e_i[i, 0], axis=1), e_i[i, 1], axis=2) for i in range (9)], axis=3)
        f_i = f_i * tf.expand_dims(spacing, 3) + temp2
        rho = tf.reduce_sum(f_i, axis=3)
        rho = tf.maximum(rho, drug)

        now = tf.reduce_sum(rho, axis = [1, 2])
        rho_l.append(now.numpy())
        rho_a.append((now + accu).numpy())

        f_eq = equilibrium(rho)

        if t % 10 == 9:
            print('Iter: {} Runtime: {:.5f} sec'.format(t+1, time.time() - start))
            start = time.time() 

    return rho.numpy(), rho_l, rho_a, filled


if __name__ == '__main__':
    
    Iter = 250000
    
    a = time.time()    
    rho, rho_left, rho_accu, _ = run(Iter, 0, args.n)
    b = time.time()
    
    print('Total Runtime: {} sec'.format(b-a))
    
    rho_diff = []
    for i in range (Iter):
        rho_diff.append(rho_accu[i+1] - rho_accu[i])
    
    rho_left = np.array(rho_left)
    rho_accu = np.array(rho_accu)
    rho_diff = np.array(rho_diff)
    
    np.save(path+'rho.npy', rho)
    np.save(path+'rho_left.npy', rho_left)
    np.save(path+'rho_accu.npy', rho_accu)
    np.save(path+'rho_diff.npy', rho_diff)
    
    print('Saved !')


