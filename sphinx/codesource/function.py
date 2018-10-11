"""
functions for generating binary variables
"""
import numpy as np

#=========================================================================================
""" np.sign(0) = 0 but here to avoid value 0, we redefine it as
    def sign(0) = 1
"""
def sign(x):
    return 1. if x >= 0 else -1.

#=========================================================================================
def sign_vec(x):
    x_vec = np.vectorize(sign)
    return x_vec(x)

#=========================================================================================
""" cross_covariance
a,b -->  <(a - <a>)(b - <b>)>  (axis=0) 
"""
def cross_cov(a,b):
   da = a - np.mean(a, axis=0)
   db = b - np.mean(b, axis=0)
   return np.matmul(da.T,db)/a.shape[0]


