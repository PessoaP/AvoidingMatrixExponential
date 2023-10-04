import numpy as np
import pandas as pd
import smn
from basis import *
from scipy.linalg import expm

from models import lam_1S as get_lam

@njit
def rhoGk(rho,mat_exp,k):
    rho = np.dot(rho,mat_exp)
    sep = k[1:]-k[:-1]
    rho_t = np.zeros((k.size,rho.size))
    rho_t[0]+=rho
    for i in range(len(sep)):
        for j in range(sep[i]): 
            rho = np.dot(rho,mat_exp)
        rho_t[1+i]+=rho
    return rho_t

#@njit
def take_points(pe,w_all,T_ind):
    return pe[T_ind,w_all] 



class params:
    def __init__(self,theta,N):
        birth,d = 1.0*theta
        self.value=theta
        
        rho = make_initial(0,0,N)
        self.rho = rho
                
        self.N = N
        self.lam = get_lam(birth,d,N)
        self.A = smn.get_A(self.lam)
        self.dt = 1/(self.A.values.max())

        self.log_prior = lprior(theta)
        
        
    def likes(self,T_unique):
        rho = self.rho*1.0
        step = T_unique[0]
        target_expo = np.round(T_unique/step,1).astype(int)
        
        if (T_unique/step - target_expo).max()>1e-11:
            warnings.warn('Apparently your time collection is not in a grid. This may lead to incorrect inference with matrix exponential.')
        A = self.A.to_dense()
        mat_exp = expm(step*A)
        
        return rhoGk(rho,mat_exp,target_expo)
        

    

    def loglike_w(self,w_all,T_all):
        T_unique,T_ind = np.unique(T_all,return_inverse=True)
        pe_theta = self.likes(T_unique)
        return np.log(take_points(pe_theta,w_all,T_ind))

    
def update_S(th_list):
    return (np.cov(np.log(th_list).T,bias=True) + np.eye(2)*1e-12)*((2.38/np.sqrt(2))**2) 

def update_th(ll,w_all,T_all,th,S_prop):
    value_prop = np.exp(np.random.multivariate_normal(np.log(th.value),S_prop))
    th_prop = params(value_prop,th.N)
    ll_prop = th_prop.loglike_w(w_all,T_all)

    update = np.log(np.random.rand()) < ll_prop.sum() - ll.sum() + th_prop.log_prior -th.log_prior
    if update:
        return ll_prop,th_prop
    return ll,th
    
def save_me(ll_list,th_list,beta_gt):
    ll_pd = np.stack(ll_list)
    th_pd = np.stack(th_list)

    df = pd.DataFrame(th_pd,columns=['birth rate','death rate'])
    df['log p(w|th)'] = ll_pd

    df.to_csv('inference/ME_inference_beta={}.csv'.format(beta_gt),index=False)