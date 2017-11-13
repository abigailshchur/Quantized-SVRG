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
    #return 2.0*xi*(np.dot(s*wi, xi) - yi)/s
    return None

def quantize(x, s):
	return None


def lp_sgd_lr(alpha, x, y, iters):
	return None

