import numpy as np
import pandas as pd
import smn
from basis import *

from models import lam_1S as get_lam


@njit
def rho_Bk(rho_init,B,k_all,w_all):
    k_all_indices = np.argsort(k_all) 
    sorted_k_all = k_all[k_all_indices]
    sorted_w_all = w_all[k_all_indices]

    rho = rho_init
    k = 0
    
    rhoBk = np.zeros(k_all.size)
    
    for (ind,k_t,w_t) in zip(k_all_indices,sorted_k_all,sorted_w_all):
        if k<k_t:
            for i in range(k_t-k):
                rho = smn.dot(rho,B)
                k+=1
        rhoBk[ind] = rho[w_t]
    return rhoBk

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

    def sample_k(self,T_all):
        return np.random.poisson(self.omega*T_all)
    
    def loglike_k(self,k_all,T_all):
        return loglike_poisson(k_all,self.omega*T_all)
    
    def loglike_w_k(self,w_all,k_all):
        return np.log(rho_Bk(self.rho,self.B,k_all,w_all))
          

        
def update_k(llw,llk,k_all,w_all,T_all,th):
    k_all_prop = th.sample_k(T_all)
    #k_all_prop = np.abs(k_all + np.round(np.random.normal(0,th.omega*T_all/25)).astype(int))

    llw_prop = th.loglike_w_k(w_all,k_all_prop)
    
    update = np.log(np.random.rand(w_all.size)) < llw_prop-llw
    
    res_k = arr_replace(k_all,k_all_prop,update)
    res_llw = arr_replace(llw,llw_prop,update)
    res_llk = th.loglike_k(res_k,T_all)
    
    return res_llw,res_llk,res_k

def update_S(th_list):
    return (np.cov(np.log(th_list).T,bias=True) + np.eye(2)*1e-12)*((2.38/np.sqrt(2))**2) 

def update_th(llw,llk,k_all,w_all,T_all,th,S_prop):
    value_prop = np.exp(np.random.multivariate_normal(np.log(th.value),S_prop))
    th_prop = params(value_prop,th.N)
    llw_prop = th_prop.loglike_w_k(w_all,k_all)
    llk_prop = th_prop.loglike_k(k_all,T_all)

    update = np.log(np.random.rand()) < llw_prop.sum() + llk_prop.sum() + th_prop.log_prior -llw.sum() - llk.sum() -th.log_prior
    if update:
        return llw_prop,llk_prop,th_prop
    return llw,llk,th

def save_ieu(llw_list,llk_list,th_list,beta_gt):
    llw_pd = np.stack(llw_list)
    llk_pd = np.stack(llk_list)
    th_pd = np.stack(th_list)

    df = pd.DataFrame(th_pd,columns=['birth rate','death rate'])
    df['log p(w|k,th)'] = llw_pd
    df['log p(k|th)'] = llk_pd

    df.to_csv('inference/IEU_inference_beta={}.csv'.format(beta_gt),index=False)
