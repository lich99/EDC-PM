#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:33:49 2020

@author: chenghaili
"""

import os
os.chdir(os.path.abspath(os.path.join(os.getcwd(), "..")))  

import args
import porespy
import numpy as np
from matplotlib import  pyplot as plt

# Init args
n = 1
BD = 10
size = args.size
drug = args.drug
drug_size = [size[0]-2*BD, size[1]-2*BD, size[2]-2*BD]
drug_quantity = args.drug_quantity
porosity = args.porosity
blobiness = args.blobiness
path = args.path


for i in range (n):
    
    porous = porespy.generators.blobs(size, porosity, blobiness).astype(np.float32())
    
    if drug == True:
        
            drug_filled = np.zeros(size, dtype = bool)
            drug_filled[BD:-BD, BD:-BD, BD:-BD] = porespy.generators.blobs(drug_size, drug_quantity, blobiness)
            porous[drug_filled] = 2
            
    plt.imshow(porous[:, :, 0])