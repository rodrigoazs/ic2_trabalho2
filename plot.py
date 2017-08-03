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

n_exps = 1000

x = np.arange(80, 130 + 1) #np.linspace(0, 130, num=50) #np.arange(20,70)
y = np.linspace(0, 2.5, num=51) #np.arange(-2,2)
X, Y = np.meshgrid(x, y)

p = [[0 for i in range(X.shape[0])] for j in range(X.shape[1])]
M = [[0 for i in range(X.shape[0])] for j in range(X.shape[1])]

#M[10][20] = -2
folder = 'plot'
plot_last = 5

for file_i in range(plot_last):
    with open(folder + "/stochastic_noise_"+ str(file_i+1) +".txt", "r") as file:
        for line in file:
            data = ast.literal_eval(line)
            try:
                w = p[data[0]][data[1]]
                measure = data[2] - data[3]
                M[data[0]][data[1]] = (w * n_exps * M[data[0]][data[1]] + n_exps * measure) / ((w + 1) * n_exps)
                #M[data[0]][data[1]] = data[2] - data[3]
                p[data[0]][data[1]] += 1
            except:
                print(str(data[0]) + ' ' + str(data[1]))
 
LML = np.array(M).T

plt.pcolor(X, Y, LML, vmin=-0.2, vmax=0.2, cmap='jet')
plt.colorbar()
plt.show()