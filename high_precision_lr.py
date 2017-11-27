import numpy as np
import random
import time

### This file assumes that all inputs are floating point ####


"""
Full precision gradient calculation for linear regression
"""
def gradient(wi, xi, yi):
	return 2.0*xi*(np.dot(wi, xi) - yi)

"""
Full precision stochastic gradient descent
"""
def sgd(alpha, x, y, iters):
	n,d = np.shape(x)
	wi = np.random.rand(d)
	for i in range(iters):
		idx = random.randint(0, n-1)
		xi = x[idx, :]
		yi = y[idx]
		grad = gradient(wi,xi,yi)
		wi = wi - alpha*(grad)
	return wi

"""
Full precision SVRG calculation
"""
def svrg(alpha, x, y, K, T, calc_loss = False):
	time_array = []
	loss_array = []
	n,d = np.shape(x)
	w_last = np.ones(d)*(-1)
	w_last[0] = 5/(2.0*128.0*128.0)
	for k in range(K):
		w_tilde = w_last
		mu_tilde = np.zeros(d)
		# calculating full gradient
		for i in range(n):
			xi = x[i, :]
			mu_tilde += gradient(w_tilde, xi, y[i])
		mu_tilde = mu_tilde/(1.0*n)
		w0 = w_tilde
		start = time.time()
		for t in range(T):
			i = random.randint(0,n-1)
			xi = x[i, :]
			yi = y[i]
			grad_1 = gradient(w0, xi, yi)
			grad_2 = gradient(w_tilde, xi, yi)
			w0 = w0 - alpha*(grad_1 - grad_2 + mu_tilde)
		w_last = w0
		time_array.append(time.time() - start)
		if (calc_loss):
			loss_array.append(loss(w_last, x, y))
	return w_last, time_array, loss_array

"""
Squared loss function
"""
def loss(w_hat, x, y):
	n,d = np.shape(x)
	loss = 0
	for i in range(n):
		xi = x[i, :]
		loss += (y[i] - (np.dot(xi.T,w_hat)))**2
	loss = loss/(1.0*n)
	return loss