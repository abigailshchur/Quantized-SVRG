import numpy as np
import random


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

def fp_svrg_lr(alpha, x, y, T):
	n,d = np.shape(x)
	w0_tilde = np.random.rand(d)*10 #init weights
	m = 2*n
	for t in range(T):
		w_tilde = w0_tilde
		mu_tilde = np.zeros(d)
		for i in range(n):
			xi = x[i, :]
			mu_tilde += gradient_lr(w_tilde, xi, y[i])
		mu_tilde = mu_tilde/(1.0*n)
		w0 = w_tilde
		for i in range(m):
			i = random.randint(0,n-1)
			xi = x[i, :]
			yi = y[i]
			grad_1 = gradient_lr(w0, xi, yi)
			grad_2 = gradient_lr(w_tilde, xi, yi)
			w0 = w0 - alpha*(grad_1 - grad_2 + mu_tilde)
		w0_tilde = w0
	return w0_tilde 

def lp_gradient_lr(wi, xi, yi, s):
    return 2.0*xi*(np.dot(s*wi, xi) - yi)

def quantize(w, s):
	w = w / s
	w2 = np.zeros(len(w), dtype=np.int8)
	# TODO optimize
	for i in range(len(w)):
		if w[i] > 127:
			w2[i] = np.int8(127)
		elif w[i] < -128:
			w2[i] = np.int8(-128)
		else:
			w2[i] = np.int8(w[i])
	return w2


def lp_sgd_lr(alpha, x, y, iters):
	n,d = np.shape(x)
	# lp weight rep, range -128 to 127
	wi = np.array( np.random.rand(d)*500, dtype=np.int8) # fix this later
	s = 1/128.0 # scale factor, makes weight range -1 to 127/128
	for i in range(iters):
		idx = random.randint(0, n-1)
		xi = x[idx, :]
		yi = y[idx]
		#print(type(wi))
		grad = lp_gradient_lr(wi,xi,yi,s)
		wi = quantize(s*wi - alpha*(grad), s)
	return wi

def lp_svrg_lr(alpha, x, y, T):
	n,d = np.shape(x)
	w0_tilde = np.array( np.random.rand(d)*500, dtype=np.int8) # fix this later
	s = 1/128.0
	m = 2*n
	for t in range(T):
		w_tilde = w0_tilde
		mu_tilde = np.zeros(d)
		for i in range(n):
			xi = x[i, :]
			mu_tilde += gradient_lr(s*w_tilde, xi, y[i])
		mu_tilde = mu_tilde/(1.0*n)
		w0 = w_tilde
		for i in range(m):
			i = random.randint(0,n-1)
			xi = x[i, :]
			yi = y[i]
			grad_1 = lp_gradient_lr(w0, xi, yi, s)
			grad_2 = lp_gradient_lr(w_tilde, xi, yi, s)
			w0 = quantize(s*w0 - alpha*(grad_1 - grad_2 + mu_tilde), s)
		w0_tilde = w0
	return w0_tilde 

