import numpy as np
import pandas as pd
import smn
from basis import *

from models import lam_1S as get_lam


@njit
def solve(rho_init,B,omegaT):
    rho = rho_init
    log_pf = (-omegaT) #poisson factor log
    pf_cum = np.exp(log_pf)

    ans = rho*np.exp(log_pf)
    lim = omegaT+6*np.sqrt(omegaT)
    for k in np.arange(1,lim):
        log_pf  += np.log(omegaT/k)
        pf = np.exp(log_pf)
        pf_cum += pf

        rho = smn.dot(rho,B)
        ans += rho*pf

    ans += rho*(1.0-pf_cum)
    return ans

def take_points(pe,w_all,T_ind): #ideally there must be a way to turn the previous and this into a single one
    return pe[T_ind,w_all] 

class params:
    def __init__(self,theta,N):
        birth,d = 1.0*theta
        self.value=theta
        
        rho = make_initial(0,0,N)
        self.rho = rho
                
        self.N = N
        self.lam = get_lam(birth,d,N)
        self.B,self.omega = smn.get_B(self.lam)
        
        self.log_prior = lprior(theta) #fix when applying to real data
        
        
    def likes(self,T_unique):
        rho = self.rho*1.0
        rho_t = []
        t=0
        for T in T_unique:
            rho = solve(rho,self.B,(T-t)*self.omega)
            rho_t.append(rho)
            t=T
        return np.vstack(rho_t)
    

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

def save(ll_list,th_list,beta_gt):
    ll_pd = np.stack(ll_list)
    th_pd = np.stack(th_list)

    df = pd.DataFrame(th_pd,columns=['birth rate','death rate'])
    df['log p(w|th)'] = ll_pd

    df.to_csv('inference/IMU_inference_beta={}.csv'.format(beta_gt),index=False)