import numpy as np
import random
import time

"""
Low precision (8 bit) gradient calculation for linear regression
(does not multiple by xi, just returns a scalar)
"""
def gradient(wi, xi, sx, sw, yi):
	dp = np.dot(wi.astype(float)*sw,xi.astype(float)*sx)
	return 2.0*(dp - yi)

def gradient_full(wi, xi, yi):
	return 2.0*xi*(np.dot(wi, xi) - yi)


def quantize(vec, s, qtype, vmin, vmax):
	vec = vec/s
	vec=np.asfarray(vec)
	vec[vec > vmax] = vmax
	vec[vec < vmin] = vmin
	Qvec=np.array(vec, dtype=qtype)
	Qvec[vec%1>np.random.rand()]+=1
	return Qvec


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
def svrg(w0, alpha, x, y, K, T, calc_loss = False):
	time_array = []
	loss_array = []
	n,d = np.shape(x)
	sb = 1/((1.0)*(2**7)); sr = 1/(1.0*2**7); sg = 1/(1.0*2**15); sp = sg/sr# defining all scales
	max16 = 1.0*((2**15)-1); min16 = -1.0*((2**15))
	max8 = 1.0*((2**7)-1); min8 = -1.0*((2**7))
	bb = 8; br = 8; bp = 8; bg = 16
	w_last = quantize(w0,sb,np.int8, min8, max8)
	loss_array.append(loss(w_last.astype(float)*sb, x.astype(float)*sr, y.astype(float)))
	for k in range(K):
		print("== > start epoch", k+1)
		check=time.time()
		w_tilde = w_last
		mu_tilde = np.zeros(d) # full precision for now
		for i in range(n):
			xi = x[i, :]
			g=gradient(w_tilde, xi, sr, sb, y[i])
			grd = g*(xi.astype(float)*sr)
			mu_tilde += grd # add full precision gradient
		print("Compute gradient -- Time cost: ", time.time()-check)
		check=time.time()
		print("Begin quantize to green")
		mu_tilde = quantize(alpha*mu_tilde/(1.0*n), sg, np.int16, min16, max16)
		print("Finished quantize to green, Time cost", time.time()-check)
		w = w_tilde
		start = time.time()
		print("Start epoch ", k, " inner iter")
		check=time.time()
		time_grad=0
		time_quan=0
		time_dot=0
		time_sum=0
		for t in range(T):
			i = random.randint(0,n-1)
			xi = x[i, :]
			yi = y[i]
			checkInner=time.time()
			grad1 = gradient(w, xi, sr, sb, yi) # fp gradient wrt w
			grad2 = gradient(w_tilde, xi, sr, sb, yi) #fp gradient wrt w_tilde
			time_grad+=time.time()-checkInner
			#grad = alpha*(grad1-grad2)
			grad = (grad1-grad2)
			checkInner=time.time()
			temp1 = quantize(grad, sp, np.int8, min8, max8)
			time_quan+=time.time()-checkInner
			temp1 = temp1.astype(float)*sp
			#temp1 = temp1.astype(np.int16)
			#x16= xi.astype(np.int16)
			checkInner=time.time()
			#temp2 = np.dot(temp1, x16)
			temp2 = np.dot(temp1, xi.astype(float)*sr)
			time_dot+=time.time()-checkInner
			checkInner=time.time()
			#sum1=np.array(temp2, dtype=np.int32)  + np.array(mu_tilde, dtype=np.int32)
			#sum2=np.array(w, dtype=np.int32)
			#w = np.left_shift(sum2,8) - sum1
			w = np.array(w, dtype=float)*sb - alpha*temp2*sg - np.array(mu_tilde, dtype=float)*sg
			time_sum+=time.time()-checkInner
			checkInner=time.time()
			#w = quantize(w.astype(float)*sg,sb,np.int8, min8, max8)
			w = quantize(w.astype(float),sb,np.int8, min8, max8)
			time_quan+=time.time()-checkInner
		w_last = w
		time_array.append(time.time() - start)
		print("Finished epoch ", k+1, " inner iter, Total time cost ", time.time()-check)
		print("     Time cost for gradident: ", time_grad)
		print("     Time cost for quantization: ", time_quan)
		print("     Time cost for dot product: ", time_dot)
		print("     Time cost for summing: ", time_sum)
		print("-----------------------------------------------")
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
    loss= np.average((y - x.dot(w_hat)) ** 2, axis=0)
    return loss
