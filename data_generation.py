import numpy as np
import random

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