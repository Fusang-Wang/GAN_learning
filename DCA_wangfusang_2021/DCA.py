# 2021.1.4 wang fusang email:1290391034@qq.com

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import DCD

class DCA():
    ## difference of convex decomposition algorithm for neural networks
    def __init__(self,model,gradients,loss,rho_rate=0.01,rho=0.2,x0=None):
        # parameters for algorithm
        self.rho = rho
        self.rho_small = False # for futur use to mark when rho is two small to have approx_loss > loss
        self.rho_rate = rho_rate # change rate when rho is too small
        
        # expected parameters from the network
        self.model = model
        self.gradients = gradients
        self.loss = loss
        self.x0 = x0
        
        # copy some of the given parameters form network for protection since python uses, in general, call for reference
        # further improvement: use copy whe necessary to safe memory
        self.variables = model.trainable_variables
        self.g = self.copy(self.gradients)
        self.v = self.copy(model.trainable_variables)

################### opration designed for tensorlist ################  
    def copy(self,tensor_list):
        res = list(tensor_list)
        for i in range(len(tensor_list)):
            res[i] = tensor_list[i].numpy().copy()
        return res
      

    def assign(self,tensor_list,xk_1):
        for i in range(len(tensor_list)):
            tensor_list[i].assign(xk_1[i])
            
########################### end #####################################
    
    def approx_loss(self):
        
        if len(self.v)==len(self.g):
            self.length = len(self.variables)
        
        if self.x0 is None: 
            self.xk_0 = self.copy(self.variables)
            self.xk_1 = self.copy(self.variables)   
        else:
            self.xk_0 = self.copy(self.x0)
            self.xk_1 = self.copy(self.x0) 
            
        if (len(self.v)==len(self.g)):
            approx_loss = []
            dcd_list = []
            for i in range(self.length):

                ### tensor operations to solve the convex optimization problem ###
                #a = tf.multiply(self.g[i], -1/self.rho)
                
                #b = tf.add(self.v[i], a)
                
                self.xk_1[i]=self.xk_0[i] - 1/self.rho*self.g[i]

                dcd = DCD.DCD(self.xk_0[i],self.xk_1[i],self.gradients[i],1,self.loss)
                layer_approx_L = dcd.loss_app.numpy()
            
                approx_loss.append(layer_approx_L)
                dcd_list.append
            approx_L = np.array(approx_loss).sum()

            self.assign(self.variables,self.xk_1)# store the calculated solution to the convex problem to xk_1
            return approx_L
# futher improvement: adjust rho when too small, or accelerate the change rate of rho when necessary 
    def apply_gradient(self,approx_L,loss_xk_1):
        # the approximaited convex solved problem is only meaningful when approx_loss > loss, thus we should update the obtained wieght to the netwaork carefully 
        # if the approximated loss at xk_1 is smaller than th ereal loss at xk_1, the weight shall be changed back to the original one xk_0 and rho shall be bigger 
        if (approx_L < loss_xk_1):
            print("rho too small")
            self.assign(self.variables,self.xk_0)
            self.rho = self.rho+self.rho_rate
            print("current rho",self.rho)
        
# further improvement: reset rho when it reaches a certain value