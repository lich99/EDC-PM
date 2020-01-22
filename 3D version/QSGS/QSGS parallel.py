# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 00:28:27 2020

@author: Chenghai Li
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:30:37 2020

@author: Chenghai Li
"""
import time
import numpy as np
import tensorflow as tf
from numpy import pi, cos, sin
from numpy.random import uniform
from matplotlib import pyplot as plt


# Params
L_0 = 50      # Length of side
D_1 = 0.4    # Total Drug ratio
a_0 = 6       # Elliptic Param
b_0 = a_0/2   # Elliptic Param
c_0 = b_0     # Elliptic Param
D_2 = 0.5     # Total Soild ratio
cdd = 0.01    # Soild core ratio
bd = 5       # Boundary size
d_list = [0.2] * 6 + [0.05] * 12  
# Probability of growth in 18 directions
dtype = tf.float32

def wex3d(n, cdd, M):
    
    global d_list

    lx, ly, lz = M.shape
    
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
    
    M_expand = np.zeros((lx+2, ly+2, lz+2))
    M_expand[1:-1, 1:-1, 1:-1] = M
    M_expand = tf.convert_to_tensor(M_expand, dtype = dtype)
    
    bound = np.ones((lx+2, ly+2, lz+2))
    bound[0, :, :] = 0
    bound[-1, :, :] = 0
    bound[:, 0, :] = 0
    bound[:, -1, :] = 0
    bound[:, :, 0] = 0
    bound[:, :, -1] = 0

    bound = tf.convert_to_tensor(bound, dtype = dtype)
    
    while temp_num_soild < num_total_need:
                                                                 
        M_static = tf.cast(M_expand == 1, dtype = dtype)
        added = tf.zeros((lx+2, ly+2, lz+2), dtype = tf.int32)
        for i in range(18):
            prob = tf.random.uniform((lx+2, ly+2, lz+2))
            add_check = tf.cast(prob < d_list[i], dtype = dtype)
            add = tf.roll(tf.roll(tf.roll(M_static, e_i[i, 0], axis=0), e_i[i, 1], axis=1), e_i[i, 2], axis=2)
            ready_to_add = add * bound * add_check
            added = tf.maximum(added, tf.cast((ready_to_add - M_expand) == 1, tf.int32))
            M_expand = tf.maximum(M_expand, ready_to_add)
            
        temp_num_soild = tf.reduce_sum(tf.cast(M_expand == 1, dtype = dtype))
        print(temp_num_soild.numpy(), num_total_need)
    
    exceeded = int(temp_num_soild.numpy() - num_total_need)
    index = np.nonzero(added.numpy())
    # print(exceeded, index[0].shape[0])
    choice = np.random.choice(index[0].shape[0], exceeded, replace=False)
    M_expand = M_expand.numpy()
    for i in range (exceeded):
        M_expand[index[0][choice[i]], index[1][choice[i]], index[2][choice[i]]] = 0
    # print(np.sum(M_expand==1))
    return M_expand[1:-1, 1:-1, 1:-1], time_in

def drug3d(n, a, b, c, bd, M):
    
    L = M.shape[0]
    total_d = (L ** 3) * n
    number_seed = 0 
    temp_d = 0
    
    while temp_d < total_d:
       
        x_temp, y_temp, z_temp = np.floor((L - 2*bd) * uniform(size = [3])).astype(np.int) + bd
        while M[x_temp, y_temp, z_temp] == 2:
            x_temp, y_temp, z_temp = np.floor((L - 2*bd) * uniform(size = [3])).astype(np.int) + bd
            
        number_seed += 1 
            
        alpha, beta, gamma = pi * uniform(size = [3])
     
        Alpha = np.array([[1,          0,           0],
                          [0, cos(alpha), -sin(alpha)],
                          [0, sin(alpha),  cos(alpha)]], dtype = np.float32)
        
        Beta = np.array([[cos(beta), 0, -sin(beta)],
                         [0,         1,          0],
                         [sin(beta), 0,  cos(beta)]], dtype = np.float32)
        
        Gamma = np.array([[cos(gamma), -sin(gamma), 0],
                          [sin(gamma),  cos(gamma), 0],
                          [         0,           0, 1]], dtype = np.float32)
    
        mat = Gamma.dot(Beta).dot(Alpha)
        r = max(a, b, c)
        for i in range (max(x_temp - r, 0), min(x_temp + r+1, L)):
            for j in range (max(y_temp - r, 0), min(y_temp + r+1, L)):
                for k in range (max(z_temp - r, 0), min(z_temp + r+1, L)):
                    Temp = mat.dot([[i-x_temp],
                                    [j-y_temp],
                                    [k-z_temp]])
                    
                    if ((Temp[0][0]/a)**2 + (Temp[1][0]/b)**2 + (Temp[2][0]/c)**2) <= 1:
                    # if np.sum(np.square(Temp[:, 0]/[a, b, c])) <= 1:
                        M[i,j,k] = 2   
                        
        temp_d = np.sum(M)/2
        
    return M
     
M = np.zeros((L_0, L_0, L_0))
time0 = time.time()
M = drug3d(D_1, a_0, b_0, c_0, bd, M)
time1 = time.time()
print('Part 1 Runtime: {:5f} sec'.format(time1-time0))
M, timein = wex3d(D_2,cdd,M)
time2 = time.time()
print('Part 2 Runtime: {:5f} sec'.format(timein-time1))
print('Part 3 Runtime: {:5f} sec'.format(time2-timein))

'''

for i in range (L_0):
    plt.clf()
    plt.imshow(M[:, :, i])
    plt.savefig(str(i)+'.png')
'''
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
 
x = np.arange(0, L_0)
y = np.arange(0, L_0)
x, y = np.meshgrid(x, y)
for i in range (10):
    z = np.ones((L_0, L_0)) * i
    y3 = M[:, :, i]
    ax.scatter(x, y, z,c=y3.flatten(), marker='s', s=50,linewidths=1, depthshade = False, alpha = 0.15, label='')
plt.show()
'''

