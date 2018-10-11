"""
inferring quadratic interactions
"""
import numpy as np
from scipy import linalg
import function as ft

#=========================================================================================
"""generate quadratic term with standard deviation g/n then make it to be 
   symmetry: q[i,j,k] = q[i,k,j] and no-self interactions: q[i,j,j] = 0  
""" 
def generate_quadratic(g,n):
    q = np.random.normal(0.0,g/n,size=(n,n,n))
    
    # make q to be symmetry and no-self interaction
    for j in range(n):
        for k in range(n):
            if k > j: q[:,j,k] = q[:,k,j]
            if k == j: q[:,j,j] = 0.
    return q 
#=========================================================================================
"""generate binary time-series 
    input: interaction matrix w[n,n], interaction variance g, data length l
    output: time series s[l,n]
""" 
def generate_data(w,q,l):
    n = np.shape(w)[0]

    s = np.ones((l,n))
    for t in range(1,l-1):    
        # calculate h1[t,i] = sum_j w[i,j]*s[t,j]
        h1 = np.sum(w[:,:]*s[t,:],axis=1) # Wij from j to i

        # calculate h2[t,i] = sum q[i,j,k]*s[t,j]*s[t,k]
        h2 = np.einsum('ijk,j,k->i', q,s[t,:],s[t,:])

        h = h1 + 0.5*h2

        p = 1/(1+np.exp(-2*h))
        s[t+1,:]= ft.sign_vec(p-np.random.rand(n))

    return s
#=========================================================================================
""" inference interactions (including linear term Wij and quadratic term Qijk from observations)
"""
def inference(s):
    l,n= np.shape(s)

    m = np.mean(s[:-1],axis=0)
    ds = s[:-1] - m

    c = np.cov(ds,rowvar=False,bias=True)
    c1 = linalg.inv(c) # inverse
    W = np.empty((n,n)) ; Q = np.empty((n,n,n))

    nloop = 100
    
    for i0 in range(n):
        s1 = s[1:,i0]
        h = s1
        cost = np.full(nloop,100.)
        for iloop in range(nloop):
            dh = h - h.mean()

            ##-------- calculate q1-------------
            ## q1[j,k] = sum_{mu,nu} <dh[t]*dss[t,mu,nu]>_t * c1[j,mu] * c1[k,nu] 

            ## ss[t,i,j] = s[t,i]*s[t,j]
            ss = s[:-1,:,np.newaxis]*s[:-1,np.newaxis,:]
            dss = ss - np.mean(ss,axis=0)

            ##dhdss[t,mu,nu] = dh[t]*dss[t,mu,nu]:
            dhdss = dh[:,np.newaxis,np.newaxis]*dss
            dhdss_av = np.mean(dhdss,axis=0)

            q11 = np.dot(c1,dhdss_av)
            q1 = np.dot(q11,c1.T)

            ##-------- calculate q2-------------
            ## dsdss[t,lamda,mu,nu] = ds[t,lamda]*dss[t,mu,nu]
            dsdss = ds[:,:,np.newaxis,np.newaxis]*dss[:,np.newaxis,:,:]

            ## dsdss_av[lamda,mu,nu] = <dsdss[t,lamda,mu,nu]>_t 
            dsdss_av = np.mean(dsdss,axis=0)

            ## q22[j,k,l] = dsdss_av[lamda,mu,nu]*c1[j,lamda]*c1[k,mu]*c1[l,nu]:
            q22 = np.einsum(dsdss_av,[3,4,5],c1,[0,3],c1,[1,4],c1,[2,5],[0,1,2])

            ## dhds_av = <dh[t]*ds[t,i]>_t:
            dhds = dh[:,np.newaxis]*ds
            dhds_av = np.mean(dhds,axis=0)

            ## q2 = sum_l dhds_av[l]*q22[j,k,l]:
            q2 = np.einsum('l,jkl->jk',dhds_av,q22)

            ##-------- calculate q = q1 - q2 -------------
            q = q1 - q2

            ## q[j,j] = 0.
            np.fill_diagonal(q,0.)

            w = np.dot(c1,dhds_av) - np.dot(q,m)

            h = np.dot(s[:-1,:],w[:]) + 0.5*np.einsum('jk,tj,tk->t', q,s[:-1],s[:-1])

            s_model = np.tanh(h)

            cost[iloop] = np.mean((s1[:]-s_model[:])**2)

            if cost[iloop] >= cost[iloop-1]: break

            h *= np.divide(s1,s_model, out=np.ones_like(s1), where=s_model!=0)

            #mse = np.mean((w0[i0,:]-w[:])**2)
            #slope = np.sum(w0[i0,:]*w)/np.sum(w0[i0,:]**2) 

            #mse_q = np.mean((q0[i0,:,:]-q)**2)
            #slope_q = np.sum(q0[i0,:,:]*q)/np.sum(q0[i0,:,:]**2) 

            #print(slope,slope_q,cost[iloop])
        W[i0] = w ; Q[i0] = q
    return W,Q
