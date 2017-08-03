# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:18:33 2017

@author: Rodrigo Azevedo
"""

import matplotlib.pyplot as plt
import numpy as np
import ast

def stochastic_noise(N_bound, sigma2_bound, sigma2_samples):
    x = np.arange(N_bound[0], N_bound[1] + 1)
    y = np.linspace(sigma2_bound[0], sigma2_bound[1], num=sigma2_samples)
    return [x, y]

def deterministic_noise(N_bound, Qf_bound,):
    x = np.arange(N_bound[0], N_bound[1] + 1)
    y = np.arange(Qf_bound[0], Qf_bound[1] + 1)
    return [x, y]

def plot(values, total_plots, filename):
    X, Y = np.meshgrid(values[0], values[1])

    p = [[0 for i in range(X.shape[0])] for j in range(X.shape[1])]
    M = [[0 for i in range(X.shape[0])] for j in range(X.shape[1])]
    
    folder = 'plot'    
    for file_i in range(total_plots):
        with open(folder + "/" + filename + "_" + str(file_i+1) +".txt", "r") as file:
            for line in file:
                data = ast.literal_eval(line)
                try:
                    w = p[data[0]][data[1]]
                    measure = data[2] - data[3]
                    M[data[0]][data[1]] = (w *  M[data[0]][data[1]] + measure) / (w + 1)
                    p[data[0]][data[1]] += 1
                except:
                    print(str(data[0]) + ' ' + str(data[1]))
     
    LML = np.array(M).T
    
    plt.pcolor(X, Y, LML, vmin=-0.2, vmax=0.2, cmap='jet')
    plt.colorbar()
    plt.show()

#plot(stochastic_noise([80,130], [0.0, 2.5], 51), 5, 'stochastic_noise')
plot(deterministic_noise([1,130], [0, 100]), 1, 'deterministic_noise')