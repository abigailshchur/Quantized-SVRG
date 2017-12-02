# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 13:26:39 2017

"""


import numpy as np

out_iter = 5000
in_iter = 10

n=50
d=20
M = np.random.rand(n,d)
l,v,r = np.linalg.svd(M)
v_full = np.append(np.diag(v), np.zeros((n-d,d)), axis=0)
RM = np.dot(np.dot(l,v_full), r)
print(np.linalg.norm(M - RM, ord='fro'))

# rank 2 approx
rank = 2
v2 = [v[i] if i < rank else 0 for i in range(len(v))]
v2_full = np.append(np.diag(v2), np.zeros((n-d,d)), axis=0)
RM2 = np.dot(np.dot(l,v2_full), r)
print(np.linalg.norm(M - RM2, ord='fro'))

# rank 3 approx
rank = 3
v3 = [v[i] if i < rank else 0 for i in range(len(v))]
v3_full = np.append(np.diag(v3), np.zeros((n-d,d)), axis=0)
RM3 = np.dot(np.dot(l,v3_full), r)
print(np.linalg.norm(M - RM3, ord='fro'))

# set half of the values to 0
mask = np.random.randint(0,2,size=M.shape).astype(np.bool)
z = np.zeros((n,d))
RM3_zero = RM3.copy()
RM3_zero[mask] = z[mask]

def matrix_factorization_SGD(R, P, Q, K, steps=5000, alpha=0.00001, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR =  np.dot(P,Q)
        #e = 0
        #for i in xrange(len(R)):
        #    for j in xrange(len(R[i])):
        #        if R[i][j] > 0:
        #            e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
        #            for k in xrange(K):
        #                e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        #if (steps > 4900):
        #    print(e)
        #if e < 0.001:
        #    print("converged")
        #    break
    return P, Q.T

def matrix_factorization_SVRG(R, P, Q, K, steps=5000, alpha=0.00001, beta=0.02):
    Q = Q.T
    
    P_last = np.random.normal(scale=1./K, size= P.shape)
        
    Q_last = np.random.normal(scale=1./K, size= Q.shape)
    
    for _ in range(out_iter):
        P_tilde = P_last
            
        Q_tilde = Q_last
        
        P_mu_tilde = np.zeros(shape =P.shape)
        
        Q_mu_tilde = np.zeros(shape =Q.shape)
        
        for i in range(len(R)):
            for j in range(len(R[i])):
                for k in range(K):
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    
                    P_mu_tilde[i][k] = P_mu_tilde[i][k]  + 2 * eij * Q_tilde[k][j] - beta * P_tilde[i][k]
                    
                    Q_mu_tilde[k][j] = Q_mu_tilde[k][j] + 2 * eij * P_tilde[i][k] - beta * Q_tilde[k][j]
                    
        P_mu_tilde = P_mu_tilde/K
        
        Q_mu_tilde = Q_mu_tilde/K
        for step in range(steps):
            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                        for k in range(K):
                            P[i][k] = P[i][k] + alpha * ((2 * eij * Q[k][j] - beta * P[i][k]) -(2 * eij * Q_tilde[k][j] - beta * P_tilde[i][k]) + P_mu_tilde[i][k])
                            
                            Q[k][j] = Q[k][j] + alpha * ((2 * eij * P[i][k] - beta * Q[k][j]) -
                             (2* eij * P_tilde[i][k] - beta * Q_tilde[k][j]) + Q_mu_tilde[k][j])
            
        P_last = P
        Q_last = Q
        eR =  np.dot(P,Q)

        return P, Q.T
    
K=3
P = np.random.rand(n,K)
Q = np.random.rand(d,K)
nP, nQ = matrix_factorization_SVRG(RM3_zero, P, Q, K, steps=50000)

# for 50000 iterations
RM3_est = np.dot(nP,nQ.T)
RM3_est_z = RM3_est.copy()
RM3_est_z[mask] = z[mask]
print(np.linalg.norm(RM3_est_z - RM3_zero, ord='fro'))
print(np.linalg.norm(RM3_est - RM3, ord='fro'))

# for 5000 iterations
RM3_est = np.dot(nP,nQ.T)
RM3_est_z = RM3_est.copy()
RM3_est_z[mask] = z[mask]
print(np.linalg.norm(RM3_est_z - RM3_zero, ord='fro'))
print(np.linalg.norm(RM3_est - RM3, ord='fro'))


