# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:08:50 2017

@author: Rodrigo Azevedo
"""

import time
import sys
import numpy as np
import math
from ofexp import OverfittingExp
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial import legendre as Legendre

class OExp1:
    def __init__(self, Qf, N, sigma2):
        self.Qf = Qf # polynomial degree
        self.N = N # dataset size
        self.sigma2 = sigma2 # noise level
        
    def generate_target(self):
        coefs = np.random.randn(self.Qf + 1)
        s = 0.0
        for coef in coefs:
            s += coef**2
        coefs = np.divide(coefs, np.sqrt(s))
        self.target = Legendre.Legendre(coefs)
        
    def generate_dataset(self):
        sigma = np.sqrt(self.sigma2)
        X = np.random.uniform(-1.0, 1.0, self.N)
        y = self.target(X) + (sigma * np.random.randn(self.N))
        self.dataset = [X, y]
        
    def generate_g2(self):
        self.g2 = Polynomial.fit(self.dataset[0], self.dataset[1], 2, [-1.0, 1.0])
        self.g2_eout = self.eout(self.g2)
        self.g2_eout2 = self.eout2(self.g2)
        self.g2_eout3 = self.eout3(self.g2)
        #self.g2_eout4 = self.eout4(self.g2)
        #self.g2_eout5 = self.eout5(self.g2)
        
    def generate_g10(self):
        self.g10 = Polynomial.fit(self.dataset[0], self.dataset[1], 10, [-1.0, 1.0])
        self.g10_eout = self.eout(self.g10)
        self.g10_eout2 = self.eout2(self.g10)
        self.g10_eout3 = self.eout3(self.g10)
        #self.g10_eout4 = self.eout4(self.g10)
        #self.g10_eout5 = self.eout5(self.g10)
    
    # sqrt(int((f(x)-g(x))^2))
    def eout(self, g):
        # definite integral from -1 to 1
        target_poly = self.target.convert(kind=Polynomial)
        comp = (g - target_poly)**2
        comp_integ = comp.integ()
        return np.sqrt(comp_integ(1.0) - comp_integ(-1.0))
    
    # sqrt(int((f(x)-g(x))^2)/int(1))
    def eout2(self, g):
        # definite integral from -1 to 1
        target_poly = self.target.convert(kind=Polynomial)
        comp = (g - target_poly)**2
        comp_integ = comp.integ()
        return np.sqrt((comp_integ(1.0) - comp_integ(-1.0))/2)
    
    # int((f(x)-g(x))^2)/int(1)
    def eout3(self, g):
        # definite integral from -1 to 1
        target_poly = self.target.convert(kind=Polynomial)
        comp = (g - target_poly)**2
        comp_integ = comp.integ()
        return (comp_integ(1.0) - comp_integ(-1.0))/2
    
    # numerical mse
    def eout4(self, g):
        # definite integral from -1 to 1
        error = 0.0
        n = 10000
        target_poly = self.target.convert(kind=Polynomial)
        x_space = np.linspace(-1.0, 1.0, num=n)
        for x in x_space:
            error += (g(x) - target_poly(x))**2
        return error / n
    
    # numerical rmse
    def eout5(self, g):
        # definite integral from -1 to 1
        error = 0.0
        n = 10000
        target_poly = self.target.convert(kind=Polynomial)
        x_space = np.linspace(-1.0, 1.0, num=n)
        for x in x_space:
            error += (g(x) - target_poly(x))**2
        return np.sqrt(error / n)

    def run(self):
        self.generate_target()
        self.generate_dataset()
        self.generate_g2()
        self.generate_g10()

def display_time(seconds, granularity=2):
    result = []
    intervals = (
    ('w', 604800),  # 60 * 60 * 24 * 7
    ('d', 86400),    # 60 * 60 * 24
    ('h', 3600),    # 60 * 60
    ('m', 60),
    ('s', 1),
    )

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])

g2_eouts = np.zeros((1000, 3))
g10_eouts = np.zeros((1000, 3))  

total_tasks = 1000
tasks_count = 0
last_time = 0
start_time = time.time()

for i in range(1000):
    a = OExp1(20,120,2)
    a.run()
    g2_eouts[i] = np.array([a.g2_eout, a.g2_eout2, a.g2_eout3])
    g10_eouts[i] = np.array([a.g10_eout, a.g10_eout2, a.g10_eout3])
    
    tasks_count += 1
    last_time = time.time()
    exec_time = last_time - start_time
    remaining_time = (total_tasks - tasks_count) * (exec_time) / tasks_count
    sys.stdout.write("\rCalculado ... %.2f%%. Tempo execução: %s. Tempo restante estimado: %s" % (((100.0 * tasks_count / total_tasks)), display_time(last_time - start_time), display_time(remaining_time)))
    sys.stdout.flush()

g2_mean = np.mean(g2_eouts, axis=0)
g10_mean = np.mean(g10_eouts, axis=0)

print(g10_mean - g2_mean)