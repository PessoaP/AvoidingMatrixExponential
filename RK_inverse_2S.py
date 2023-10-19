import numpy as np
import pandas as pd
import smn
from basis import *

from models import lam_2S as get_lam

@njit
def evolve_RK(rho,A,dt):
    k1=smn.dot(rho,A)
    k2=smn.dot(rho + k1*(dt/2),A)
    k3=smn.dot(rho + k2*(dt/2),A)
    k4=smn.dot(rho + k3*dt,A)
    return rho + (dt/6)*(k1+k2+k2+k3+k3+k4)

@njit
def solve(rho_0,A,T,dt):
    rho=1.0*rho_0
    t=0*dt
    while t+dt<T:
        t+=dt
        rho = evolve_RK(rho,A,dt)
    return evolve_RK(rho,A,T-t)

#@njit
def take_points(pe,w_all,T_ind,N_RNA): #ideally there must be a way to turn the previous and this into a single one
    return pe[T_ind,w_all] + pe[T_ind,w_all+N_RNA]

class params:
    def __init__(self,theta,N_RNA):
        N_DNA = 2
        birth,d,l01,l10 = 1.0*theta
        self.value=theta
        
        rho = make_initial(0,0,N_RNA*N_DNA)##this creates on inactive state
        self.rho = rho
        
        self.N_RNA = N_RNA
        self.N = N_RNA*N_DNA
        self.lam = get_lam(birth,d,l01,l10,N_RNA,N_DNA)
        self.A = smn.get_A(self.lam)
        self.dt = 1/(np.abs(self.A.values).max())

        self.log_prior = lprior(theta)
        
        
    def likes(self,T_unique):
        rho = self.rho*1.0
        rho_t = []
        t=0
        for T in T_unique:
            rho = solve(rho,self.A,T-t,self.dt)
            rho_t.append(rho)
            t=T
        return np.vstack(rho_t)

    def loglike_w(self,w_all,T_all):
        T_unique,T_ind = np.unique(T_all,return_inverse=True)
        pe_theta = self.likes(T_unique)
        return np.log(take_points(pe_theta,w_all,T_ind,self.N_RNA))

    
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

def save(ll_list,th_list,beta_gt,burnin=False):
    ll_pd = np.stack(ll_list)
    th_pd = np.stack(th_list)

    df = pd.DataFrame(th_pd,columns=['birth rate','death rate','activation rate','deactivtion rate'])
    df['log p(w|th)'] = ll_pd

    filename = '2S_RK_inference_beta={}.csv'.format(beta_gt)
    folder ='inference/'
    if burnin:
        folder='burn/'

    df.to_csv(folder+filename,index=False)