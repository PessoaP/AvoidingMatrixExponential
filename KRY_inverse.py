import numpy as np
import pandas as pd
import smn
from basis import *
from scipy.linalg import expm

from models import lam_1S as get_lam


@njit
def kryreconstruct(Q,exp_H,rho):
    exp_col = np.ascontiguousarray(exp_H[:,0])
    return normalize(np.dot(Q,exp_col)*np.linalg.norm(rho))

def evolve_KRY(rho,A,dt,kappa):
    Q, H = arnoldi(A, rho, kappa)
    exp_H = expm(dt*H)
    return kryreconstruct(Q,exp_H,rho)


def solve(rho_0,A,T,dt,kappa=20):
    rho=1.0*rho_0
    t=0*dt
    while t+dt<T:
        t+=dt
        rho = evolve_KRY(rho,A,dt,kappa)
    return evolve_KRY(rho,A,T-t,kappa)

#@njit
def take_points(pe,w_all,T_ind): #ideally there must be a way to turn the previous and this into a single one
    return pe[T_ind,w_all] 

class params:
    def __init__(self,theta,N,kappa=20):
        birth,d = 1.0*theta
        self.value=theta
        
        rho = make_initial(0,0,N)
        self.rho = rho
                
        self.N = N
        self.lam = get_lam(birth,d,N)
        self.A = smn.get_A(self.lam)
        self.dt = 1.5*kappa/(np.abs(self.A.values).max())
        self.kappa = kappa

        self.log_prior = lprior(theta)
        
        
    def likes(self,T_unique):
        rho = self.rho*1.0
        rho_t = []
        t=0
        for T in T_unique:
            rho = solve(rho,self.A,T-t,self.dt,self.kappa)
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

    df.to_csv('inference/KRY_inference_beta={}.csv'.format(beta_gt),index=False)