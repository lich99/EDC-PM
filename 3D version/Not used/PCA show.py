#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 22:17:53 2020

@author: chenghaili
"""
import numpy as np
from matplotlib import  pyplot as plt

plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

a = np.load('0.5.npy')
b = np.load('1.npy')
c = np.load('2.npy')

plt.plot(a)
plt.plot(b)
plt.plot(c)
