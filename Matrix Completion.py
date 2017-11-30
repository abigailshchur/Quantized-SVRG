# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 01:30:57 2017

@author: Sergio
"""

import numpy as np

import time

import random


O = 100 # no. of outer iterations

I = 3 # no. inner iterations

class MF():
    
    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        
        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """
        
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
        
        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        
        self.no_samples = len(self.samples)
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))
        
        return training_process
    
    def train2(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
        
        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        
        self.no_samples = len(self.samples)
        
        # Perform stochastic gradient descent for number of iterations
        
        training_process = []
        
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.svrg()
            mse = self.mse()
            training_process.append((i, mse))
        
        if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))
        
        return training_process
    
    
    

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic gradient descent
        """
        
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            
            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])
            
    
            
            
            
    def svrg(self):
        """
        Perform SVRG
        
        """
        time_array = [] 
        
        b_u_last = np.random.normal(scale=1./self.K, size= self.b_u.shape)
        
        b_i_last = np.random.normal(scale=1./self.K, size= self.b_i.shape)
        
        P_last = np.random.normal(scale=1./self.K, size= self.P.shape)
        
        Q_last = np.random.normal(scale=1./self.K, size= self.Q.shape)
        
        
        
        for _ in range(O):
            
            b_u_tilde = b_u_last
            
            b_i_tilde = b_i_last
            
            P_tilde = P_last
            
            Q_tilde = Q_last
            
            b_u_mu_tilde = np.zeros(shape =self.b_u.shape)
            
            b_i_mu_tilde = np.zeros(shape =self.b_i.shape)
        
            P_mu_tilde = np.zeros(shape =self.P.shape)
        
            Q_mu_tilde = np.zeros(shape =self.Q.shape)
            
            for i, j, r in self.samples:
                
                prediction = self.get_rating(i, j)
                
                e = r - prediction
            
                b_u_mu_tilde[i] += e - self.beta * b_u_tilde[i]
        
                b_i_mu_tilde[j] += e - self.beta * b_i_tilde[j]
                
                self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
                
                self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])
                
                P_tilde[i,:]
                
                Q_tilde[j, :]
                
                P_mu_tilde[i, :]

                P_mu_tilde[i, :] += e * Q_tilde[j, :] - self.beta * P_tilde[i,:]
        
                Q_mu_tilde[j, :] += e * P_tilde[i, :] - self.beta * Q_tilde[j,:]
                
            b_u_mu_tilde = b_u_mu_tilde/self.no_samples
        
            b_i_mu_tilde = b_i_mu_tilde/self.no_samples
        
            P_mu_tilde = P_mu_tilde/self.no_samples
        
            Q_mu_tilde = Q_mu_tilde/self.no_samples
        
            b_u_0 = b_u_tilde
        
            b_i_0 = b_i_tilde
        
            P_0 = P_tilde
        
            Q_0 = Q_tilde
        
            start = time.time()
            
            for _ in range(I):
                k = random.randint(0,self.no_samples - 1)
            
                sample = self.samples[k]
                
                i, j, _ = sample
                
                b_u_grad1 =   self.alpha * (e - self.beta * b_u_0[i])
            
                b_i_grad1 = self.alpha * (e - self.beta * b_i_0[j])
        
                P_grad1 = self.alpha * (e * Q_0[j, :] - self.beta * P_0[i,:])
            
                Q_grad1 = self.alpha * (e * P_0[i, :] - self.beta * Q_0[j,:])
    
                b_u_grad2 =   self.alpha * (e - self.beta * b_u_tilde[i])
            
                b_i_grad2 = self.alpha * (e - self.beta * b_i_tilde[j])
        
                P_grad2 = self.alpha * (e * Q_tilde[j, :] - self.beta * P_tilde[i,:])
            
                Q_grad2 = self.alpha * (e * P_tilde[i, :] - self.beta * Q_tilde[j,:])
                
                b_u_0 = b_u_0 - self.alpha*(b_u_grad1 - b_u_grad2 + b_u_mu_tilde)
                
                b_i_0 = b_i_0 - self.alpha*(b_i_grad1 - b_i_grad2 + b_i_mu_tilde)
            
                P_0 = P_0 - self.alpha*(P_grad1 - P_grad2 + P_tilde)
            
                Q_0 = Q_0 - self.alpha*(Q_grad1 - Q_grad2 + Q_tilde)
                
        b_u_last = b_u_0
        
        b_i_last = b_i_0
        
        P_last = P_0
        
        Q_last = Q_0
        
        time_array.append(time.time() - start)

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction
    
    
    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)
    
    
x = MF(R,K, alpha, beta, iterations)

x.train2()