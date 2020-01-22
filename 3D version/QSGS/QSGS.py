# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:30:37 2020

@author: Chenghai Li
"""
import time
import numpy as np
from math import pi, cos, sin
from numpy.random import uniform
from matplotlib import pyplot as plt


# 参数设定
L_0 = 50
D_1 = 0.1 # 药物体积比
a_0 = 8 # 椭圆参数
b_0 = a_0/2
c_0 = b_0
D_2 = 0.6 # 胶质占比
cdd = 0.01 # 胶质数量密度


def wex3d(n, cdd, M):

    lx, ly, lz = M.shape
    
    d_list = [0.8] * 6 + [0.1] * 12
    
    num_total_need = n * lx * ly *lz                                               
    num_soild = 0                                                     
    soild = []
    
    e_i = np.zeros((19, 3), dtype = np.int32)
    e_i[0]  = [ 1, 0, 0]
    e_i[1]  = [-1, 0, 0]
    e_i[2]  = [ 0, 0, 1]
    e_i[3]  = [ 0, 0,-1]
    e_i[4]  = [ 0,-1, 0]
    e_i[5]  = [ 0, 1, 0]
    e_i[6]  = [ 1, 0, 1]
    e_i[7]  = [-1, 0, 1]
    e_i[8]  = [ 1, 0,-1]
    e_i[9]  = [-1, 0,-1]
    e_i[10] = [ 1,-1, 0]
    e_i[11] = [-1,-1, 0]
    e_i[12] = [ 1, 1, 0]
    e_i[13] = [-1, 1, 0]
    e_i[14] = [ 0,-1, 1]
    e_i[15] = [ 0,-1,-1]
    e_i[16] = [ 0, 1, 1]
    e_i[17] = [ 0, 1,-1]
    
    for i in range (lx):  
    	for j in range (ly):
            for k in range (lz): 
                if uniform() < cdd: 
                    if M[i,j,k] != 2:
                        num_soild += 1 
                        M[i,j,k] = 1 
                        soild.append([i, j, k])
    
    temp_num_soild = num_soild
    
    time_in = time.time()
    
    while temp_num_soild < num_total_need:
                                                                 
        for index_soild in range (temp_num_soild):
            
            index = soild[index_soild]
            check = uniform(size = [18]) < d_list
            
            for i in range(18):
                if check[i] == True:
                    x_new = index[0] + e_i[i][0]
                    y_new = index[1] + e_i[i][1]
                    z_new = index[2] + e_i[i][2]
                    if 0 <= x_new < lx and 0 <= y_new < ly and 0 <= z_new < ly:
                        if M[x_new, y_new, z_new] == 0:
                            num_soild += 1
                            M[x_new, y_new, z_new] = 1
                            soild.append([x_new, y_new, z_new]) 
    
        temp_num_soild = num_soild
    print(temp_num_soild)
    return M, time_in

def drug3d(n, a, b, c, M):
    
    L = M.shape[0]
    total_d = (L ** 3) * n
    number_seed = 0 
    temp_d = 0
    
    while temp_d < total_d:
       
        x_temp, y_temp, z_temp = np.floor(L * uniform(size = [3])).astype(np.int)
        while M[x_temp, y_temp, z_temp] == 2:
            x_temp, y_temp, z_temp = np.floor(L * uniform(size = [3])).astype(np.int)
            
        number_seed += 1 
            
        alpha, beta, gamma = pi * uniform(size = [3])
     
        Alpha = np.array([[1,          0,           0],
                          [0, cos(alpha), -sin(alpha)],
                          [0, sin(alpha),  cos(alpha)]])
        
        Beta = np.array([[cos(beta), 0, -sin(beta)],
                         [0,         1,          0],
                         [sin(beta), 0,  cos(beta)]])
        
        Gamma = np.array([[cos(gamma), -sin(gamma), 0],
                          [sin(gamma),  cos(gamma), 0],
                          [         0,           0, 1]])
    
        r = max(a, b, c)
        for i in range (max(x_temp - r, 0), min(x_temp + r+1, L)):
            for j in range (max(y_temp - r, 0), min(y_temp + r+1, L)):
                for k in range (max(z_temp - r, 0), min(z_temp + r+1, L)):
                    Temp = Gamma.dot(Beta).dot(Alpha).dot([[i-x_temp],
                                                           [j-y_temp],
                                                           [k-z_temp]])
                    
                    if ((Temp[0][0]/a)**2 + (Temp[1][0]/b)**2 + (Temp[2][0]/c)**2) <= 1:
                        M[i,j,k] = 2   
                        
        temp_d = np.sum(M)
        
    return M
     
M = np.zeros((L_0, L_0, L_0))
time0 = time.time()
M = drug3d(D_1, a_0, b_0, c_0, M)
time1 = time.time()
print('Part 1 Runtime: {:5f} sec'.format(time1-time0))
M, timein = wex3d(D_2,cdd,M)
time2 = time.time()
print('Part 1.5 Runtime: {:5f} sec'.format(timein-time1))
print('Part 2 Runtime: {:5f} sec'.format(time2-time1))


