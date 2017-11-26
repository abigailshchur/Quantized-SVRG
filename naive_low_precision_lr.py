import numpy as np
import random
import time


def gradient_lr(wi, xi, yi):
    return 2.0*xi*(np.dot(wi, xi) - yi)

def fp_sgd_lr(alpha, x, y, iters):
	n,d = np.shape(x)
	wi = np.random.rand(d)
	for i in range(iters):
		idx = random.randint(0, n-1)
		xi = x[idx, :]
		yi = y[idx]
		grad = gradient_lr(wi,xi,yi)
		wi = wi - alpha*(grad)
	return wi


def lp_gradient_lr(wi, xi, yi, s):
    return 2.0*xi*(np.dot(s*wi, xi) - yi)

def quantize(vec, s, qtype, vmin, vmax):
	vec = vec / s
	vec2 = np.zeros(len(vec), dtype=qtype)
	for i in range(len(vec)):
		#print(vec[i])
		#print(vmax)
		#print(vmin)
		if vec[i] > vmax:
			#print("here1")
			vec2[i] = qtype(vmax)
		elif vec[i] < vmin:
			#print("here2")
			vec2[i] = qtype(vmin)
		else:
			#print("here3")
			vec2[i] = qtype(vec[i])
			if np.random.rand() > vec[i]%1 and vec2[i] < vmax:
				vec2[i]+=1
	return vec2

"""
Naive Implementation of LP SVRG
"""
def lp_svrg_lr(alpha, x, y, T, calc_loss = True):
	time_array = []
	loss_array = []
	n,d = np.shape(x)
	w0_tilde = np.array(np.random.rand(d)*32768, dtype=np.int16) # fix this later
	w0_tilde = w0_tilde.astype(float)
	s = 1/(2.0*128.0*128.0)
	m = 2*n
	for t in range(T):
		w_tilde = w0_tilde
		mu_tilde = np.zeros(d)
		for i in range(n):
			xi = x[i, :]
			#print(np.shape(xi))
			#print(np.shape(s*w_tilde))
			#print(np.shape(y[i]))
			#print(y[i])
			mu_tilde += gradient_lr(s*w_tilde, xi, y[i])
		mu_tilde = alpha*mu_tilde/(1.0*n)
		w0 = w_tilde
		start = time.time()
		for i in range(m):
			i = random.randint(0,n-1)
			xi = x[i, :]
			yi = y[i]
			grad_1 = lp_gradient_lr(w0, xi, yi, s)
			grad_2 = lp_gradient_lr(w_tilde, xi, yi, s)
			w0 = quantize(s*w0 - alpha*grad_1 + alpha*grad_2 - mu_tilde, s, np.int16, -32768, 32767)
			w0 = w0.astype(float)
		w0_tilde = w0
		time_array.append(time.time() - start)
		if (calc_loss):
			loss_array.append(loss(w0_tilde.astype(float)*s, x.astype(float), y.astype(float)))
	return w0_tilde*s, time_array, loss_array

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
