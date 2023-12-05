import RK_inverse_STS as rk
import KRY_inverse_STS as kry
import RMJP_inverse_STS as rmjp

import pandas as pd
import time

import numpy as np 
from matplotlib import pyplot as plt

import sys

method = sys.argv[3]
methods = {
    'rk': rk,
    'kry': kry,
    'rmjp': rmjp,
}
met = methods.get(method)
if not met:
    print('method not recognized')

N_sam = 100*20+1
np.random.seed(200)


beta_R_gt = float(sys.argv[1])
beta_P_gt = float(sys.argv[2])

l01_gt = .1
l10_gt = .05

gamma_R_gt = 1.
gamma_P_gt = .1


df = pd.read_csv('synthetic_data/STS_synthetic_data--beta={}-{}.csv'.format(beta_R_gt,beta_P_gt))

w_all = df[['counts_P','counts_R']].to_numpy().astype(int)
T_all = df['times'].to_numpy()
NP,NR = 2*w_all.max(axis=0)
N = 2*NP*NR

S_prop = np.eye(6)*1e-8


ground = np.array((beta_R_gt,beta_P_gt,l01_gt,l10_gt,gamma_R_gt,gamma_P_gt))
th_gt = met.params(ground,NP,NR)

ll_gt = th_gt.loglike_w(w_all,T_all)

theta = ground
th = met.params(theta,NP,NR)

ll = th.loglike_w(w_all,T_all)

met.update_th(ll,w_all,T_all,th,S_prop)

ll_list =[]
th_list =[]

times100 =[]


start=time.time()
for i in range(len(ll_list),N_sam):
    ll,th  = met.update_th(ll,w_all,T_all,th,S_prop)    
    ll_list.append(ll.sum())
    th_list.append(th.value)
    #print(llw,llk, llw+llk)
    
    if i%100 == 0:
        end=time.time()
        ll_last = np.stack(ll_list[-101:])
        th_last  = np.stack(th_list[-101:])

        print(ll_last.mean())
        print(th_last.mean(axis=0))
        if i>0:
            print('iteration ',i)
            times100.append(end-start)
            print('accept rate',np.mean(((th_last[1:]-th_last[:-1]).mean(axis=1)!=0)))
        print(end-start)            
        start=time.time()   


try:
    df2 = pd.read_csv('times/STS_{}_times_20.csv'.format(method.upper()))
except:
    df2 = pd.DataFrame()
label = 'N={}'.format(N)
df2[label] = np.stack(times100)
df2.to_csv('times/STS_{}_times_20.csv'.format(method.upper()),index=False)



