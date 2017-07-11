# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 16:14:26 2017

@author: Rodrigo Azevedo
"""

from multiprocessing import Pool
import random
import numpy as np
import time
import sys

def calculate_noise(rounds, N, Qf, sigma2):
    return random.uniform(-2, 2)

def do_task(args):
    return calculate_noise(*args)

def stochastic_noise(n_processes, N_bound, sigma2_bound, N_step, sigma2_step, rounds, Qf):
    N_space = np.linspace(N_bound[0], N_bound[1], num=((N_bound[1] - N_bound[0]) / (N_step*1.0) + 1.0))
    sigma2_space = np.linspace(sigma2_bound[0], sigma2_bound[1], num=((sigma2_bound[1] - sigma2_bound[0]) / (sigma2_step*1.0) + 1.0))
    tasks = []
    
    for N in N_space:
        for sigma2 in sigma2_space:
            tasks.append((rounds, N, Qf, sigma2))
            
    total_tasks = len(tasks)
    tasks_count = 0
            
    pool = Pool(n_processes)
    with open('stochastic_noise.txt', 'wb') as file:
        for result in pool.imap_unordered(do_task, tasks):
            tasks_count += 1
            file.write(str(result) + '\n')
            #print('Calculado: ' + (100.0 * tasks_count / total_tasks) + '% do experimento')
            sys.stdout.write("\rCalculado ... %s%%" % ((100.0 * tasks_count / total_tasks)))
            sys.stdout.flush()

#stochastic_noise(4, [0,10], [0,10], 2, 2, 10, 10)
a = calculate_noise(4, 1, 1,2)

n_processes= 1
N_bound=[0,2]
sigma2_bound=[0,2]
N_step=1
sigma2_step=1
rounds=1
Qf=1

N_space = np.linspace(N_bound[0], N_bound[1], num=((N_bound[1] - N_bound[0]) / (N_step*1.0) + 1.0))
sigma2_space = np.linspace(sigma2_bound[0], sigma2_bound[1], num=((sigma2_bound[1] - sigma2_bound[0]) / (sigma2_step*1.0) + 1.0))
tasks = []

for N in N_space:
    for sigma2 in sigma2_space:
        tasks.append((rounds, N, Qf, sigma2))
        
total_tasks = len(tasks)
tasks_count = 0

pool = Pool(n_processes)
#with open('stochastic_noise.txt', 'wb') as file:
    #file.write('aaa' + '\n')
for result in pool.imap_unordered(do_task, tasks):
#for t in tasks:
    result = do_task(t)
    #print(t)
    tasks_count += 1
    #file.write(str(result) + '\n')
    #print(result)
    sys.stdout.write("\rCalculado ... %s%%" % ((100.0 * tasks_count / total_tasks)))
    sys.stdout.flush()