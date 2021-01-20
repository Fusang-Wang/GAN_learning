# 2021.1.4 wang fusang email:1290391034@qq.com

import numpy as np
## The class DCD stores useful information of difference of convex decomposition : loss(x) = 1/2*rho*x^2-(1/2*rho*x^2-loss) at given point x0

class DCD():## difference of convex decomposition at a point x0


    def __init__(self,x0,x1,gradient,rho,loss):
        self.x0 =x0
        self.loss = loss
        self.rho = rho
        self.dh_x0 = rho*x0-gradient
        self.loss_app = self.loss_approx(x1)
        
    def dot(self,matrix1,matrix2): # not necessary and can be optimized by np.dot
        return (np.array(matrix1)*np.array(matrix2)).sum()
    
    def g(self,x):# let g = 1/2*rho*x^2
        return 1/2*self.rho*self.dot(x,x)
    
    def dg(self,x):# the differential of g at x
        return self.rho*x
    
    
    def h(self,x):# let h(x) = 1/2*rho*x^2 - loss
        return 1/2*self.rho*self.dot(x,x)-self.loss
        
    def h_approx(self,x):# the lineair approximation lineaire of  h on x0 h_approx = h(x0)+ dh.(x-x0)
        return self.h(self.x0) + self.dot(self.dh_x0,x-self.x0)
    
    def loss_approx(self,x):# the original non convex optimization problem become a series of convex optimization problem
        return self.g(x)-self.h_approx(x)