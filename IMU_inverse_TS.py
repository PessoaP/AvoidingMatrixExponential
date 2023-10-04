import numpy as np
import pandas as pd
import smn
from basis import *

from models import lam_TS as get_lam

    

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

#@njit
def take_points(pe,na,nb,T_ind,NA,NB): #ideally there must be a way to turn the previous and this into a single one
    ind = NB*na+nb
    return sum([pe[T_ind,ind+i*NA*NB] for i in range(4)]) #sum not numpy.sum


class params:
    def __init__(self,theta,NA,NB):
        beta_A, beta_B, gamma, l01A, l10A,l01B,l10B = 1.0*theta
        self.value=theta
        
        rho = make_initial(0,0,4*NA*NB)##this creates on inactive state
        self.rho = rho
        
        self.NA = NA
        self.NB = NB
        self.N = 4*NA*NB

        self.lam = get_lam(beta_A, beta_B, gamma,l01A,l10A,l01B,l10B,NA,NB)
        self.B,self.omega = smn.get_B(self.lam)

        self.log_prior = lprior(theta)
        
        
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
        return np.log(take_points(pe_theta,w_all[:,0],w_all[:,1],T_ind,self.NA,self.NB))

    
def update_S(th_list):
    return (np.cov(np.log(th_list).T,bias=True) + np.eye(4)*1e-12)*((2.38/np.sqrt(4))**2) 

def update_th(ll,w_all,T_all,th,S_prop):
    value_prop = np.exp(np.random.multivariate_normal(np.log(th.value),S_prop))
    th_prop = params(value_prop,th.N_RNA)
    ll_prop = th_prop.loglike_w(w_all,T_all)

    update = np.log(np.random.rand()) < ll_prop.sum() - ll.sum() + th_prop.log_prior -th.log_prior
    if update:
        return ll_prop,th_prop
    return ll,th

def save_imu(ll_list,th_list,beta_gt):
    ll_pd = np.stack(ll_list)
    th_pd = np.stack(th_list)

    df = pd.DataFrame(th_pd,columns=['birth rate','death rate','activation rate','deactivtion rate'])
    df['log p(w|th)'] = ll_pd

    df.to_csv('inference/2S_IMU_inference_beta={}.csv'.format(beta_gt),index=False)