import numpy as np
import random
import time

"""
Low precision (8 bit) gradient calculation for linear regression
(does not multiple by xi, just returns a scalar)
"""
def gradient(wi, xi, sx, sw, yi):
	dp = np.dot(wi.astype(np.int32), xi.astype(np.int32))
	return 2.0*(float(dp)*sx*sw - yi)

def gradient_full(wi, xi, yi):
	return 2.0*xi*(np.dot(wi, xi) - yi)


def quantize(vec, s, qtype, vmin, vmax):
	vec = vec/s
	vec=np.asfarray(vec)
	vec[vec > vmax] = vmax
	vec[vec < vmin] = vmin
	Qvec=np.array(vec, dtype=qtype)
	Qvec[vec%1>np.random.rand()]+=1
	Qvec=Qvec.astype(float)*s
	return Qvec


def halp(w0, alpha, x, y, K, T, mu, bit, btype, calc_loss = True):
	time_array = []
	loss_array = []
	n,d = np.shape(x)
	z0 = np.array(np.zeros(d))
	s = 1/((1.0)*(2**(bit-1)))
	w_tilde=w0
	smax=2**(bit-1)-1
	smin=-2**(bit-1)
	for t in range(K):
		w_tilde = w_tilde + z0
		mu_tilde = np.zeros(d)
		for i in range(n):
			xi = x[i, :]
			mu_tilde += gradient_full(w_tilde, xi, y[i])
		mu_tilde = mu_tilde/(1.0*n)
		s = np.linalg.norm(mu_tilde)/(mu*((2**(bit-1)))-1)
		z = np.array(np.zeros(d))
		start = time.time()
		for i in range(T):
			i = random.randint(0,n-1)
			xi = x[i, :]
			yi = y[i]
			grad_1 = gradient_full(w_tilde+z, xi, yi)
			grad_2 = gradient_full(w_tilde, xi, yi)
			z_update = z - alpha*(grad_1 - grad_2 + mu_tilde)
			z = quantize(z_update, s, btype, smin, smax)
		z0 = z
		time_array.append(time.time() - start)
		if (calc_loss):
			loss_array.append(loss(w_tilde+z, x, y))
	return w_tilde+z, time_array, loss_array

"""
Squared loss function
"""
def loss(w_hat, x, y):
    #loss = np.average((y - x.dot(w_hat)) ** 2, axis=0)
	#return loss
	loss = np.average((y - x.dot(w_hat)) ** 2, axis=0)
	return loss
