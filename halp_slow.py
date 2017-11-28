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
	vec = vec / s
	vec2 = np.zeros(len(vec), dtype=qtype)
	for i in range(len(vec)):
		if vec[i] > vmax:
			vec2[i] = qtype(vmax)
		elif vec[i] < vmin:
			vec2[i] = qtype(vmin)
		else:
			vec2[i] = qtype(vec[i])
			if np.random.rand() > vec[i]%1 and vec2[i] < vmax:
				vec2[i]+=1
	return vec2

def quantize_n(n, s, qtype, vmin, vmax):
	n = n / s
	if n > vmax:
		return qtype(vmax)
	elif n < vmin:
		return qtype(vmin)
	else:
		n2 = qtype(n)
		if np.random.rand() > n%1 and n2 < vmax:
			return n2+1
	return n2

def halp(alpha, x, y, T, calc_loss = True):
	time_array = []
	loss_array = []
	# mu = 1/3.0 # variance of x sample
	mu = 3 # variance of x sample
	n,d = np.shape(x)
	w_tilde = np.random.rand(d) # full precision
	z0 = np.array(np.zeros(d), dtype=np.int16)
	s = 1/((1.0)*(2**15))
	m = 2*n
	for t in range(T):
		w_tilde = w_tilde + z0.astype(float)*s
		mu_tilde = np.zeros(d)
		for i in range(n):
			xi = x[i, :]
			mu_tilde += gradient_full(w_tilde, xi, y[i])
		mu_tilde = mu_tilde/(1.0*n)
		s = np.linalg.norm(mu_tilde)/(mu*((2**15)-1))
		z = np.array(np.zeros(d), dtype=np.int16)
		start = time.time()
		for i in range(m):
			i = random.randint(0,n-1)
			xi = x[i, :]
			yi = y[i]
			full_z = z.astype(float)*s
			grad_1 = gradient_full(w_tilde+full_z, xi, yi)
			grad_2 = gradient_full(w_tilde, xi, yi)
			z_update = full_z - alpha*(grad_1 - grad_2 + mu_tilde)
			z = quantize(z_update, s, np.int16, -32768, 32767)
		z0 = z
		time_array.append(time.time() - start)
		if (calc_loss):
			loss_array.append(loss(w_tilde+z.astype(float)*s, x, y))
	return w_tilde+z.astype(float)*s, time_array, loss_array

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