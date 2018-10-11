##========================================================================================
import numpy as np
from scipy import linalg
from scipy.integrate import quad
from scipy.optimize import fsolve

import function as ft
""" --------------------------------------------------------------------------------------
Inferring interaction from data by Free Energy Minimization (FEM)
input: time series s
output: interaction w, local field h0
"""
def fem(s):
    l,n = np.shape(s)
    m = np.mean(s[:-1],axis=0)
    ds = s[:-1] - m
    l1 = l-1

    c = np.cov(ds,rowvar=False,bias=True)
    c_inv = linalg.inv(c)
    dst = ds.T

    W = np.empty((n,n)) #; H0 = np.empty(n)
    
    nloop = 10000

    for i0 in range(n):
        s1 = s[1:,i0]
        h = s1
        cost = np.full(nloop,100.)
        for iloop in range(nloop):
            h_av = np.mean(h)
            hs_av = np.dot(dst,h-h_av)/l1
            w = np.dot(hs_av,c_inv)
            #h0=h_av-np.sum(w*m)
            h = np.dot(s[:-1,:],w[:]) # + h0
            
            s_model = np.tanh(h)
            cost[iloop]=np.mean((s1[:]-s_model[:])**2)
                        
            if cost[iloop] >= cost[iloop-1]: break
                       
            h *= np.divide(s1,s_model, out=np.ones_like(s1), where=s_model!=0)
            #t = np.where(s_model !=0.)[0]
            #h[t] *= s1[t]/s_model[t]
            
        W[i0,:] = w[:]
        #H0[i0] = h0
    return W #,H0 


"""---------------------------------------------------------------------------------------
Inferring interaction by Maximum Likelihood Estimation (MLE)
"""
def mle(s,rate,stop_criterion): 

    l,n = s.shape
    rate = rate/l
    
    s1 = s[:-1]
    W = np.zeros((n,n))

    nloop = 10000
    for i0 in range(n):
        st1 = s[1:,i0]
        
        #w01 = w0[i0,:]    
        w = np.zeros(n)
        h = np.zeros(l-1)
        cost = np.full(nloop,100.)
        for iloop in range(nloop):        
            dw = np.dot(s1.T,(st1 - np.tanh(h)))        
            w += rate*dw        
            h = np.dot(s1,w)            
            cost[iloop] = ((st1 - np.tanh(h))**2).mean() 
                       
            if ((stop_criterion=='yes') and (cost[iloop] >= cost[iloop-1])):
                break              
        
        W[i0,:] = w
    
    return W

"""---------------------------------------------------------------------------------------
Inferring interaction from data by nMF method
input: time series s
output: interaction w
"""
def nmf(s):
    l,n = s.shape
    # empirical value:  
    m = np.mean(s,axis=0)
    
    # A matrix
    A = 1-m**2
    A_inv = np.diag(1/A)
    A = np.diag(A)
    
    # equal-time correlation:
    ds = s - m
    C = np.cov(ds,rowvar=False,bias=True)
    C_inv = linalg.inv(C)
    
    # one-step-delayed correlation:
    s1 = s[1:]
    ds1 = s1 - np.mean(s1, axis=0)
    D = ft.cross_cov(ds1,ds[:-1])    
    ##------------------
    ## predict W:
    B = np.dot(D,C_inv)
    w = np.dot(A_inv,B)
    
    ##------------------
    #MSE = np.mean((w0 - w)**2)
    #slope = np.sum(w0*w)/np.sum(w0**2)    
    #print(MSE,slope)

    return w

"""---------------------------------------------------------------------------------------
Inferring interaction from data by TAP method
input: time series s
output: interaction w
"""
def tap(s):
    n = s.shape[1]
    # nMF part: ---------------------------------------------------    
    m = np.mean(s,axis=0)
    # A matrix
    A = 1-m**2
    A_inv = np.diag(1/A)
    A = np.diag(A)

    # equal-time correlation:
    ds = s - m
    C = np.cov(ds,rowvar=False,bias=True)
    C_inv = linalg.inv(C)
    
    # one-step-delayed correlation:
    s1 = s[1:]
    ds1 = s1 - np.mean(s1, axis=0)
    D = ft.cross_cov(ds1,ds[:-1])    
    ##------------------
    ## predict W_nMF:
    B = np.dot(D,C_inv)
    w_nMF = np.dot(A_inv,B)
    #--------------------------------------------------------------

    # TAP part    
    # solving Fi in equation: F(1-F)**2) = (1-m**2)sum_j W_nMF**2(1-m**2) ==> 0<F<1
    step = 0.001
    nloop = int(0.33/step)+2

    w2_nMF = w_nMF**2
    temp = np.empty(n) ; F = np.empty(n)
    for i in range(n):
       temp[i] = (1-m[i]**2)*np.sum(w2_nMF[i,:]*(1-m[:]**2))
    
       y=-1. ; iloop=0
       while y < 0 and iloop < nloop:
          x = iloop*step
          y = x*(1-x)**2-temp[i]
          iloop += 1

       F[i] = x
    
       #F[i]=np.sqrt(temp[i])
    
    # A_TAP matrix
    A_TAP = np.empty(n)
    for i in range(n):
       A_TAP[i] = A[i,i]*(1-F[i])
    A_TAP_inv = np.diag(1/A_TAP)
    
    w_TAP = np.dot(A_TAP_inv,B)

    return w_TAP

#=========================================================================================
"""---------------------------------------------------------------------------------------
Inferring interaction from data by exact Mean Field (eMF)
input: time series s
output: interaction w
"""
def emf(s,stop_criterion):
    n = s.shape[1]
    
    # nMF part: ---------------------------------------------------    
    m = np.mean(s,axis=0)
    # A matrix
    A = 1-m**2
    #A_inv = np.diag(1/A)
    A = np.diag(A)

    # equal-time correlation:
    ds = s - m
    C = np.cov(ds,rowvar=False,bias=True)
    C_inv = linalg.inv(C)
    
    # one-step-delayed correlation:
    s1 = s[1:]
    ds1 = s1 - np.mean(s1, axis=0)
    D = ft.cross_cov(ds1,ds[:-1])    
    ##------------------
    ## predict W_nMF:
    B = np.dot(D,C_inv)
    #w_nMF = np.dot(A_inv,B)
    
    #-------------------------------------------------------------------------------------
    fun1 = lambda x,H: (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)*np.tanh(H + x*np.sqrt(delta))
    fun2 = lambda x: (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)*(1-np.square(np.tanh(H + x*np.sqrt(delta))))
    
    w_eMF = np.empty((n,n))
    
    nloop = 100

    for i0 in range(n):
        cost = np.zeros(nloop+1) ; delta = 1.

        def integrand(H):
            y, err = quad(fun1, -np.inf, np.inf, args=(H,))
            return y - m[i0]
    
        for iloop in range(1,nloop):
            H = fsolve(integrand, 0.)
            H = float(H)
    
            a, err = quad(fun2, -np.inf, np.inf)
            a = float(a)
    
            if a !=0: 
                delta = (1/(a**2))* np.sum((B[i0,:]**2) * (1-m[:]**2))
                W_temp = B[i0,:]/a

            H_temp = np.dot(s[:-1,:], W_temp)
            cost[iloop] = np.mean((s1[:,i0] - np.tanh(H_temp))**2)
    
            if ((stop_criterion=='yes') and (cost[iloop] >= cost[iloop-1])): break

        w_eMF[i0,:] = W_temp[:]

    return w_eMF
