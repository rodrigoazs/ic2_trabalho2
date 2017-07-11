# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:18:33 2017

@author: Rodrigo Azevedo
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import time

x = np.arange(30)
y = np.arange(30)
X, Y = np.meshgrid(x, y)

n = [[str(i) +'-'+str(j) for i in range(X.shape[0])] for j in range(X.shape[1])]
a = [[random.uniform(0,2) for i in range(X.shape[0])] for j in range(X.shape[1])]

a[10][20] = -2
 
LML = np.array(a).T

plt.pcolor(X, Y, LML, vmin=-0.2, vmax=0.2, cmap='jet')
plt.colorbar()
plt.show()