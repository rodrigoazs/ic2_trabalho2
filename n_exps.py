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

g2_avg_mse_eout = []
g10_avg_mse_eout = []
g2_avg_rmse_eout = []
g10_avg_rmse_eout = []

total_tasks = n_exps * n_rounds
tasks_count = 0
last_time = 0
start_time = time.time()

Qf = 20
N = 100
sigma2 = 0.05

for i in range(n_rounds):
    g2_mse_eout = []
    g10_mse_eout = []
    g2_rmse_eout = []
    g10_rmse_eout = []
    for j in range(n_exps):
        exp = OverfittingExp(Qf, N, sigma2)
        exp.run()
        g2_mse_eout.append(exp.g2_eout3)
        g2_rmse_eout.append(exp.g2_eout2)
        g10_mse_eout.append(exp.g10_eout3)
        g10_rmse_eout.append(exp.g10_eout2)
        tasks_count += 1
        last_time = time.time()
        exec_time = last_time - start_time
        remaining_time = (total_tasks - tasks_count) * (exec_time) / tasks_count
        sys.stdout.write("\rCalculado ... %.2f%%. Tempo execução: %s. Tempo restante estimado: %s" % (((100.0 * tasks_count / total_tasks)), display_time(last_time - start_time), display_time(remaining_time)))
        sys.stdout.flush()
    g2_avg_mse_eout.append((np.array(g2_mse_eout)).mean())
    g2_avg_rmse_eout.append((np.array(g2_rmse_eout)).mean())
    g10_avg_mse_eout.append((np.array(g10_mse_eout)).mean())
    g10_avg_rmse_eout.append((np.array(g10_rmse_eout)).mean())

g2_avg_mse_eout = np.array(g2_avg_mse_eout)
g10_avg_mse_eout = np.array(g10_avg_mse_eout)
dif_mse_eout = g10_avg_mse_eout - g2_avg_mse_eout
g2_avg_rmse_eout = np.array(g2_avg_rmse_eout)
g10_avg_rmse_eout = np.array(g10_avg_rmse_eout)
dif_rmse_eout = g10_avg_rmse_eout - g2_avg_rmse_eout

print('\n')
print('Teste para Qf='+str(Qf)+', N='+str(N)+', sigma2='+str(sigma2))
print('Numero de cenarios: '+str(n_exps))
print('Numero de rodadas: '+str(n_rounds))
print('\n')
print('g2 avg MSE eout')
print(g2_avg_mse_eout)
print('Desvio padrao: '+str(g2_avg_mse_eout.std()))
print('g10 avg MSE eout')
print(g10_avg_mse_eout)
print('Desvio padrao: '+str(g10_avg_mse_eout.std()))
print('Diferenca dos modelos (g10 - g2)')
print(dif_mse_eout)
print('Desvio padrao: '+str(dif_mse_eout.std()))
print('\n')
print('g2 avg RMSE eout')
print(g2_avg_rmse_eout)
print('Desvio padrao: '+str(g2_avg_rmse_eout.std()))
print('g10 avg RMSE eout')
print(g10_avg_rmse_eout)
print('Desvio padrao: '+str(g10_avg_rmse_eout.std()))
print('Diferenca dos modelos (g10 - g2)')
print(dif_rmse_eout)
print('Desvio padrao: '+str(dif_rmse_eout.std()))