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
def sgd(w0, alpha, x, y, iters):
	n,d = np.shape(x)
	wi = w0
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
def svrg(w0, alpha, x, y, K, T, calc_loss = False):
	time_array = []
	loss_array = []
	n,d = np.shape(x)
	w_last = w0
	loss_array.append(loss(w_last, x, y))
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
    #loss = np.average((y - x.dot(w_hat)) ** 2, axis=0)
	#return loss
	loss = np.average((y - x.dot(w_hat)) ** 2, axis=0)
	return loss
