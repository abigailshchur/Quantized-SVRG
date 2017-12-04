import numpy as np
import random
import time


def gradient_lr(wi, xi, yi):
    return 2.0*xi*(np.dot(wi, xi) - yi)

def lp_gradient_lr(wi, xi, yi, s):
    return 2.0*xi*(np.dot(s*wi, xi) - yi)


def quantize(vec, s, qtype, vmin, vmax):
	vec = vec/s
	vec=np.asfarray(vec)
	vec[vec > vmax] = vmax
	vec[vec < vmin] = vmin
	Qvec=np.array(vec, dtype=qtype)
	Qvec[vec%1>np.random.rand()]+=1
	return Qvec

"""
Naive Implementation of LP SVRG
"""

def lp_svrg_lr(w0, alpha, x, y, K, T, calc_loss = True):
    s=1/(1.0*2**7)
    time_array = []
    loss_array = []
    n,d = np.shape(x)
    w0_tilde = w0
    loss_array.append(loss(w0_tilde, x, y))
    for k in range(K):
        w_tilde = w0_tilde
        mu_tilde = np.zeros(d)
        for i in range(n):
            xi = x[i, :]
            mu_tilde += gradient_lr(s*w_tilde, xi, y[i])
        mu_tilde = mu_tilde/(1.0*n)
        w0 = w_tilde
        start = time.time()
        for i in range(T):
            i = random.randint(0,n-1)
            xi = x[i, :]
            yi = y[i]
            grad_1 = lp_gradient_lr(w0, xi, yi, s)
            grad_2 = lp_gradient_lr(w_tilde, xi, yi, s)
            w0 = quantize(s*w0 - alpha*(grad_1 - grad_2 + mu_tilde), s, np.int8, -128, 127)
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
    loss= np.average((y - x.dot(w_hat)) ** 2, axis=0)
    return loss
