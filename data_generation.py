import numpy as np
import random
import precision_util as p_util

"""
Inputs:
    w - the ground truth vector
    n - the number of data points to generate
Output:
    x - n x d matrix of data
    y - n x 1 array of labels for data
"""
def generate_data_linear_regression(w, n):
    d = len(w) # dimension of data
    rn = np.random.randn(n,d) # n x d matrix samples from N(0,1)
    y_noise = np.random.randn(n) # 1 x d matrix samples from N(0,1)
    r = np.random.rand(n,d)*100.0 - 50.0 # n x d matrix samples from U(-50,50)
    x = rn + r
    y = np.dot(w,x.T) + y_noise
    return x,y

def generate_data_low_precision_linear_regression(d, n, s):
    # ground truth weights
    # w = np.random.randint(60, size=(d,1), dtype=np.int8)
    w = np.random.uniform(low=-1.0, high=127/128.0, size=(d,)) # high precision ground truth weights
    x = np.random.randint(60, size=(n,d), dtype=np.int8) # low precision data representation
    y_noise = np.random.randn(n)*0.001
    # fix for no overflow in y
    #for i in range(n):
        #print(np.dot(p_util.low_precision_to_float(w.T,s), p_util.low_precision_to_float(x[i],s).T))
    #    while(np.dot(p_util.low_precision_to_float(w.T,s), p_util.low_precision_to_float(x[i],s).T) > d):
    #        x[i] = np.random.randint(128, size=(1,d), dtype=np.int8)
    # making some weights/ data points negative
    for i in range(d):
        for j in range(n):
            if np.random.rand()<=0.5:
                x[j,i] *= -1
    x_temp = p_util.low_precision_to_float(x, s)
    y = np.ravel(np.dot(w.T, x_temp.T).T) + y_noise # y is full precision
    return np.ravel(w),x,y, x_temp
    