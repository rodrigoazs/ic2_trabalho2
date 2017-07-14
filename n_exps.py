# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:20:29 2017

@author: Rodrigo Azevedo
"""

import time
import sys
import numpy as np
from ofexp import OverfittingExp

def display_time(seconds, granularity=2):
    result = []
    intervals = (
    ('semanas', 604800),  # 60 * 60 * 24 * 7
    ('dias', 86400),    # 60 * 60 * 24
    ('horas', 3600),    # 60 * 60
    ('minutos', 60),
    ('segundos', 1),
    )

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])

n_exps = 5000
n_rounds = 10

exps_g2 = [[] for i in range(10)]
exps_g10 = [[] for i in range(10)]

total_tasks = n_exps * n_rounds
tasks_count = 0
last_time = 0
start_time = time.time()

for i in range(n_rounds):
    for j in range(n_exps):
        exp = OverfittingExp(20, 100, 0.1)
        exp.run()
        exps_g2[i].append(exp.g2_eout)
        exps_g10[i].append(exp.g10_eout)
        tasks_count += 1
        last_time = time.time()
        exec_time = last_time - start_time
        remaining_time = (total_tasks - tasks_count) * (exec_time) / tasks_count
        sys.stdout.write("\rCalculado ... %.2f%%. Tempo execução: %s. Tempo restante estimado: %s" % (((100.0 * tasks_count / total_tasks)), display_time(last_time - start_time), display_time(remaining_time)))
        sys.stdout.flush()

mean_g2 = []
mean_g10 = []
mean_sqr_g2 = []
mean_sqr_g10 = []
for i in range(n_rounds):
    a = np.array(exps_g2[i])
    mean_g2.append(a.mean())
    mean_sqr_g2.append((np.sqrt(a)).mean())
    a = np.array(exps_g10[i])
    mean_g10.append(a.mean()) 
    mean_sqr_g10.append((np.sqrt(a)).mean())
    
print('std g2 ' +str(np.array(mean_g2).std()))
print('std g10 ' +str(np.array(mean_g2).std()))
print('std sqr g2 ' +str(np.array(mean_g2).std()))
print('std sqr g10 ' +str(np.array(mean_g2).std()))