"""
functions for generating binary variables
"""
import numpy as np
import function as ft

#=========================================================================================
"""generate binary time-series 
    input: interaction matrix w[n,n], interaction variance g, data length l
    output: time series s[l,n]
""" 
def generate_data(w,l):
    n = np.shape(w)[0]
    s = np.ones((l,n))
    for t in range(1,l-1):
        h = np.sum(w[:,:]*s[t,:],axis=1) # Wij from j to i
        p = 1/(1+np.exp(-2*h))
        s[t+1,:]= ft.sign_vec(p-np.random.rand(n))
    return s

