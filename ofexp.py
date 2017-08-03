import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial import legendre as Legendre

class OverfittingExp:
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
        self.g2 = Polynomial.fit(self.dataset[0], self.dataset[1], 2, [-1.0, 1.0])
        self.g2_eout = self.eout(self.g2)
        
    def generate_g10(self):
        self.g10 = Polynomial.fit(self.dataset[0], self.dataset[1], 10, [-1.0, 1.0])
        self.g10_eout = self.eout(self.g10)
  
    def eout(self, g):
        dif =  Legendre.Legendre.cast(g) - self.target
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