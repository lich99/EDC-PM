#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:44:27 2020

@author: chenghaili
"""

import os
os.chdir(os.path.abspath(os.path.join(os.getcwd(), "..")))  

import args
import porespy
import numpy as np
from matplotlib import  pyplot as plt

def Generate(args):
    
    # Init args
    BD = 10
    n = args.n
    size = args.size
    drug_size = [size[0]-2*BD, size[1]-2*BD, size[2]-2*BD]
    drug = args.drug
    drug_quantity = args.drug_quantity
    porosity = args.porosity
    blobiness = args.blobiness
    path = args.path
    
    # Create dir
    exists = os.path.exists(path)
    if exists:
        print('File existed! Are your sure to overwrite? (y/n)')
        
        wait = input()
        if wait in ['y', 'yes', 'Y', 'Yes']:
            pass
        else:
            print('Cancelled!')
            return 0
    else:
        os.makedirs(path)  
        os.makedirs(path+'img/')  

    # Generate & Save
    data = []
    for i in range (n):
        
        porous = porespy.generators.blobs(size, porosity, blobiness).astype(np.float32())
        if drug == True:
            drug_filled = np.zeros(size, dtype = bool)
            drug_filled[BD:-BD, BD:-BD, BD:-BD] = porespy.generators.blobs(drug_size, drug_quantity, blobiness)
            porous[drug_filled] = 2
        
        data.append(porous)
        
        if args.savefig == True:
            plt.clf()
            plt.imshow(porous[:, :, 0])
            plt.savefig(path+'img/'+str(i))
            
    np.save(path+'data.npy', data)
    print('Saved {} Samples! in {}'.format(n, path+'data.npy'))
    
    return 1
    
if __name__ == '__main__':
    
    Generate(args)
