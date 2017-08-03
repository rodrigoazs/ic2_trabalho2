import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial import legendre as Legendre
from numpy import polynomial as P

class OverfittingExp:
    def __init__(self, Qf, N, sigma2):
        self.Qf = Qf # polynomial degree
        self.N = N # dataset size
        self.sigma2 = sigma2 # noise level
        
    def generate_target(self):
        coefs = np.random.randn(self.Qf + 1)
        #target = Legendre.Legendre(coefs)
        #integ = (target**2).integ()
        #exp = (integ(1.0)-integ(-1.0))/2
        #theta = 1/exp
        #target2 = np.sqrt(theta)*target
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
        #self.g2_eout2 = self.eout2(self.g2)
        #self.g2_eout3 = self.eout3(self.g2)
        
    def generate_g10(self):
        self.g10 = Polynomial.fit(self.dataset[0], self.dataset[1], 10, [-1.0, 1.0])
        self.g10_eout = self.eout(self.g10)
        #self.g10_eout2 = self.eout2(self.g10)
        #self.g10_eout3 = self.eout3(self.g10)
        
#    def eout(self):
#        dif =  Legendre.Legendre.cast(self.g2) - self.target
#        s = 0.0
#        for coef in dif.coef:
#            s += coef
#        return s
#    
#    def eout(self):
#        target_poly = self.target.convert(kind=Polynomial)
#        dif = self.g2 - target_poly
#        s = 0.0
#        for coef in dif.coef:
#            s += coef
#        return s

#    def eout2(self, g):
#        target_poly = self.target.convert(kind=Polynomial)
#        x_space = np.linspace(-1, 1, num=1000)
#        e = 0.0
#        for x in x_space:
#            e += abs(g(x) - self.target(x))
#        return e / len(x_space)   
    
    def eout(self, g):
        dif =  Legendre.Legendre.cast(g) - self.target
        coefs = dif.coef
        s = 0.0
        for n in range(len(coefs)):
            s += (coefs[n]**2) / (2*n + 1)
        return s
        
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
        #print('g2 eout:' + str(self.g2_eout))
        #print('g10 eout:' + str(self.g10_eout))