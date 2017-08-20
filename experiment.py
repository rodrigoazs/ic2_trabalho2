#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 17:27:27 2017

@author: Rodrigo Azevedo
"""

import glob
import ast
import sys
import time
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as Legendre

class OverfittingExp:
    def __init__(self, noise='stochastic', N_bound=[60,130], sigma2_bound=[0.0, 2.5], Qf_bound=[0, 100], sigma2_samples=51, Qf=20, sigma2=0.1, n_exps=1000, n_rounds=50, n_processes=0):
        self.noise = noise
        self.n_exps = n_exps
        self.n_rounds = n_rounds
        self.n_processes = n_processes
        self.N_bound = N_bound
        self.sigma2_bound = sigma2_bound
        self.Qf_bound = Qf_bound
        self.sigma2_samples = sigma2_samples
        self.Qf = Qf
        self.sigma2 = sigma2
        
    def run(self):
        if self.noise == 'stochastic':
            self.run_tasks(self.stochastic_noise(self.N_bound, self.sigma2_bound, self.sigma2_samples, self.Qf, self.n_exps, self.n_rounds), 'stochastic_noise.txt')
        elif self.noise == 'deterministic':
            self.run_tasks(self.deterministic_noise(self.N_bound, self.Qf_bound, self.sigma2, self.n_exps, self.n_rounds), 'deterministic_noise.txt')
 
    def plot_stochastic_noise(self, N_bound, sigma2_bound, sigma2_samples):
        x = np.arange(N_bound[0], N_bound[1] + 1)
        y = np.linspace(sigma2_bound[0], sigma2_bound[1], num=sigma2_samples)
        return [x, y]

    def plot_deterministic_noise(self, N_bound, Qf_bound,):
        x = np.arange(N_bound[0], N_bound[1] + 1)
        y = np.arange(Qf_bound[0], Qf_bound[1] + 1)
        return [x, y]

    def plot(self):
        if self.noise == 'stochastic':
            self.plot_fig(self.plot_stochastic_noise(self.N_bound, self.sigma2_bound, self.sigma2_samples), 'Number of Data Points, $N$', 'Noise Level, $\sigma^2$', 'stochastic_noise')
        elif self.noise == 'deterministic':
            self.plot_fig(self.plot_deterministic_noise(self.N_bound, self.Qf_bound), 'Number of Data Points, $N$', 'Target Complexity, $\Q_f$',  'deterministic_noise')
        
    def plot_fig(self, values, xlabel, ylabel, filename):
        X, Y = np.meshgrid(values[0], values[1])
    
        p = [[0 for i in range(X.shape[0])] for j in range(X.shape[1])]
        M = [[0 for i in range(X.shape[0])] for j in range(X.shape[1])]

        files = glob.glob(filename + "*.txt")
        for file_i in files:
            with open(file_i, "r") as file:
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
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        
        sets = [set(i) for i in p]
        rounds = set()
        for i in sets:
            rounds = rounds.union(i)
            
        print('Number of rounds: '+ str(sorted(list(rounds))))

    def display_time(self, seconds, granularity=2):
        result = []
        intervals = (
        ('weeks', 604800),  # 60 * 60 * 24 * 7
        ('days', 86400),    # 60 * 60 * 24
        ('hours', 3600),    # 60 * 60
        ('minutes', 60),
        ('seconds', 1),
        )
    
        for name, count in intervals:
            value = seconds // count
            if value:
                seconds -= value * count
                if value == 1:
                    name = name.rstrip('s')
                result.append("{} {}".format(value, name))
        return ', '.join(result[:granularity])
    
    def calculate_noise(self, i, j, n_exps, N, Qf, sigma2):
        sum_g2_eout = 0.0
        sum_g10_eout = 0.0
        
        for r in range(n_exps):
            exp = self.OverfittingSim(Qf, N, sigma2)
            exp.run()
            sum_g2_eout += exp.g2_eout
            sum_g10_eout += exp.g10_eout
            
        return [i, j, sum_g10_eout/n_exps, sum_g2_eout/n_exps]

    def do_task(self, args):
        return self.calculate_noise(*args)
        
    def stochastic_noise(self, N_bound, sigma2_bound, sigma2_samples, Qf, n_exps, n_rounds):
        N_space = np.arange(N_bound[0], N_bound[1] + 1)
        sigma2_space = np.linspace(sigma2_bound[0], sigma2_bound[1], num=sigma2_samples)
        tasks = []
        
        for rnd in range(n_rounds):
            for i in range(len(N_space)):
                for j in range(len(sigma2_space)):
                    tasks.append((i, j, n_exps, N_space[i], Qf, sigma2_space[j]))
                    
        return tasks

    def deterministic_noise(self, N_bound, Qf_bound, sigma2, n_exps, n_rounds):
        N_space = np.arange(N_bound[0], N_bound[1] + 1)
        Qf_space = np.arange(Qf_bound[0], Qf_bound[1] + 1)
        tasks = []
        
        for rnd in range(n_rounds):
            for i in range(len(N_space)):
                for j in range(len(Qf_space)):
                    tasks.append((i, j, n_exps, N_space[i], Qf_space[j], sigma2))
                    
        return tasks
        
    def run_tasks(self, tasks, file):
        total_tasks = len(tasks)
        tasks_count = 0
        last_time = 0
        start_time = time.time()
                
        if self.n_processes > 0:
            pool = Pool(self.n_processes)
            with open(file, 'a') as file:
                for result in pool.imap_unordered(self.do_task, tasks):
                    tasks_count += 1
                    last_time = time.time()
                    file.write(str(result) + '\n')
                    exec_time = last_time - start_time
                    remaining_time = (total_tasks - tasks_count) * (exec_time) / tasks_count
                    sys.stdout.write("\rExecuted ... %.2f%%. Execution time: %s. Expected remaining time: %s" % (((100.0 * tasks_count / total_tasks)), self.display_time(last_time - start_time), self.display_time(remaining_time)))
                    sys.stdout.flush()
        else:
            with open(file, 'a') as file:
                for task in tasks:
                    result = self.do_task(task)
                    tasks_count += 1
                    last_time = time.time()
                    file.write(str(result) + '\n')
                    exec_time = last_time - start_time
                    remaining_time = (total_tasks - tasks_count) * (exec_time) / tasks_count
                    sys.stdout.write("\rExecuted ... %.2f%%. Execution time: %s. Expected remaining time: %s" % (((100.0 * tasks_count / total_tasks)), self.display_time(last_time - start_time), self.display_time(remaining_time)))
                    sys.stdout.flush()
                
    class OverfittingSim:
        def __init__(self, Qf, N, sigma2):
            self.Qf = Qf # polynomial degree
            self.N = N # dataset size
            self.sigma2 = sigma2 # noise level
            
        def generate_target(self):
            coefs = np.random.randn(self.Qf + 1)
            s = 0.0
            for n in range(len(coefs)):
                s += (coefs[n]**2) / (2*n + 1)
            self.target = Legendre.Legendre(np.divide(coefs, np.sqrt(s)))
            
        def generate_dataset(self):
            sigma = np.sqrt(self.sigma2)
            X = np.random.uniform(-1.0, 1.0, self.N)
            y = self.target(X) + (sigma * np.random.randn(self.N))
            self.dataset = [X, y]
            
        def generate_g2(self):
            self.g2 = Legendre.Legendre.fit(self.dataset[0], self.dataset[1], 2, [-1.0, 1.0])
            self.g2_eout = self.eout(self.g2)
            
        def generate_g10(self):
            self.g10 = Legendre.Legendre.fit(self.dataset[0], self.dataset[1], 10, [-1.0, 1.0])
            self.g10_eout = self.eout(self.g10)
      
        def eout(self, g):
            dif = g - self.target
            coefs = dif.coef
            s = 0.0
            for n in range(len(coefs)):
                s += (coefs[n]**2) / (2*n + 1)
            return s
    
        def plot(self):
            target_points = self.target.linspace(10000)
            g2_points = self.g2.linspace(10000)
            g10_points = self.g10.linspace(10000)
            plt.plot(target_points[0], target_points[1], label='Target')
            plt.plot(g2_points[0], g2_points[1], color='green', label='2nd Order Fit')
            plt.plot(g10_points[0], g10_points[1], color='red', label='10th Order Fit')
            plt.plot(self.dataset[0] ,self.dataset[1], 'o', label='Data', color='white', fillstyle='full', markeredgecolor='black')
            plt.legend()
            dataset_X_bound = [min(self.dataset[0]), max(self.dataset[0])]
            dataset_y_bound = [min(self.dataset[1]), max(self.dataset[1])]
            plt.axis((dataset_X_bound[0] - 0.1, dataset_X_bound[1] + 0.1, dataset_y_bound[0] - 0.1, dataset_y_bound[1] + 0.1))
            plt.show()
            
        def run(self):
            self.generate_target()
            self.generate_dataset()
            self.generate_g2()
            self.generate_g10()