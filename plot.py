# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:18:33 2017

@author: Rodrigo Azevedo
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import ast

x = np.arange(1, 130) #np.linspace(0, 130, num=50) #np.arange(20,70)
y = np.linspace(0, 2.5, num=51) #np.arange(-2,2)
X, Y = np.meshgrid(x, y)

M = [[0 for i in range(X.shape[0])] for j in range(X.shape[1])]

#M[10][20] = -2
with open("stochastic_noise.txt", "r") as file:
    for line in file:
        data = ast.literal_eval(line)
        M[data[0]][data[1]] = data[5] - data[4]
 
LML = np.array(M).T

plt.pcolor(X, Y, LML, vmin=-0.2, vmax=0.2, cmap='jet')
plt.colorbar()
plt.show()