# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 09:47:22 2020

@author: Chenghai Li
"""


import args
import time
import numpy as np
from matplotlib import  pyplot as plt




# Init args
n = args.n
size = args.size
drug = args.drug
drug_quantity = args.drug_quantity
porosity = args.porosity
blobiness = args.blobiness
path = args.path

rho1 = np.load(path+'rho1.npy')
rho_left1 = np.load(path+'rho_left1.npy')
rho_accu1 = np.load(path+'rho_accu1.npy')
rho_diff1 = np.load(path+'rho_diff1.npy')

rho2 = np.load(path+'rho2.npy')
rho_left2 = np.load(path+'rho_left2.npy')
rho_accu2 = np.load(path+'rho_accu2.npy')
rho_diff2 = np.load(path+'rho_diff2.npy')

rho_left = np.concatenate([rho_left1, rho_left2], axis = 1)
rho_accu = np.concatenate([rho_accu1, rho_accu2], axis = 1)
rho_diff = np.concatenate([rho_diff1, rho_diff2], axis = 1)
rho = np.concatenate([rho1, rho2], axis = 0)

np.save(path+'rho_left.npy', rho_left)
np.save(path+'rho_accu.npy', rho_accu)
np.save(path+'rho_diff.npy', rho_diff)
np.save(path+'rho.npy', rho)