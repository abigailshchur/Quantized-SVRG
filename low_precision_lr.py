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

"""
Low precision stochastic gradient descent
"""
def sgd(alpha, x, y, iters):
	n,d = np.shape(x)
	wi = np.array(np.random.rand(d)*500, dtype=np.int16)
	sw = 1/(2.0*128.0*128.0)
	bw = 16
	minw, maxw = range_of_lp(sw, )
	for i in range(iters):
		idx = random.randint(0, n-1)
		xi = x[idx, :]
		yi = y[idx]
		grad = gradient(wi,xi,yi)
		wi = wi - alpha*(grad)
	return wi

"""
Low precision SVRG calculation (16 bit weights)
@param alpha : learning rate
@param x : 8 bit matrix
@param sx : scale param for x
@param y : float array

Colors:
blue - weights - (sb = 1/(2*128*128), bb = 16)
red - data (x) - (sr = 1/128, br = 8)
purple - intermediate - (sp = 1/(sg/sr), bp = 8) (within gradient calc)
green - intermediate - (sg = sb, bg = 16) (full gradient)
"""
def svrg(alpha, x, y, K, T, calc_loss = False):
	time_array = []
	loss_array = []
	n,d = np.shape(x)
	w_last = np.array(np.random.rand(d)*500, dtype=np.int16) #initialization
	#w_last = np.array(np.ones(d)*(-32768), dtype=np.int16) #initialization
	#w_last[0] = np.int16(5)
	sb = 1/((1.0)*(2**15)); sr = 1/128.0; sg = sb; sp = sg/sr # defining all scales
	max16 = 1.0*((2**15)-1); min16 = -1.0*((2**15))
	max8 = 1.0*((2**7)-1); min8 = -1.0*((2**7))
	bb = 16; br = 8; bp = 8; bg = 16
	for k in range(K):
		w_tilde = w_last
		mu_tilde = np.zeros(d) # full precision for now
		for i in range(n):
			xi = x[i, :]
			grd = gradient(w_tilde, xi, sr, sb, y[i])*(xi.astype(float)*sr)
			mu_tilde += grd # add full precision gradient
		mu_tilde = quantize(alpha*mu_tilde/(1.0*n), sg, np.int16, min16, max16)
		w = w_tilde
		start = time.time()
		#count=0
		for t in range(T):
			i = random.randint(0,n-1)
			xi = x[i, :]
			yi = y[i]
			grad1 = gradient(w, xi, sr, sb, yi) # fp gradient wrt w
			grad2 = gradient(w_tilde, xi, sr, sb, yi) #fp gradient wrt w_tilde
			temp1 = quantize_n(alpha*(grad1-grad2), sp, np.int8, min8, max8)
			temp1 = temp1.astype(np.int16)
			x16 = xi.astype(np.int16)
			temp2 = np.dot(temp1, x16)
			#if count < 10:
			#	print("low p:")
			#	print(temp2)
			#	print("high p:")
			#	print(temp2.astype(float)*sg)
			#	count+=1
			w = w.astype(float)*sb  + temp2.astype(float)*sg - mu_tilde.astype(float)*sg
			w = quantize(w, sb, np.int16, min16, max16)
		w_last = w
		time_array.append(time.time() - start)
		if (calc_loss):
			loss_array.append(loss(w_last.astype(float)*sb, x.astype(float)*sr, y.astype(float)))
	return w_last.astype(float)*sb, time_array, loss_array


def range_of_lp(s,b):
	minv = -s*2**(b-1)
	maxv = s*((2**(b-1))-1)
	return (minv, maxv)

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