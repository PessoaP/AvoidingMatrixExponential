import RK_inverse as rk
import KRY_inverse as kry
import RMJP_inverse as rmjp
import ME_inverse as me

import pandas as pd
import time

import numpy as np 
from matplotlib import pyplot as plt

import sys

method = sys.argv[2]
methods = {
    'rk': rk,
    'kry': kry,
    'rmjp': rmjp,
    'me': me,
}
met = methods.get(method)
if not met:
    print('method not recognized')

N_sam = 100*20
np.random.seed(12)

beta_gt = int(sys.argv[1])
gamma_gt=1

df = pd.read_csv('synthetic_data/synthetic_data--beta={}.csv'.format(beta_gt))
w_all = (df['counts']).to_numpy(int)
T_all = (df['times']).to_numpy()
N = 2*w_all.max()

S_prop = np.eye(2)*1e-8


ground = np.array((beta_gt,gamma_gt))
th_gt = met.params(ground,N)

ll_gt = th_gt.loglike_w(w_all,T_all)


beta,gamma = met.naive_estimation(T_all,w_all)

theta = np.array((beta,gamma))
th_met = met.params(theta,N)

th = th_met
ll = th.loglike_w(w_all,T_all)

met.update_th(ll,w_all,T_all,th,S_prop)


# **Restarting chain**


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
    df2 = pd.read_csv('times/{}_times_20.csv'.format(method.upper()))
except:
    df2 = pd.DataFrame()
label = 'N={}'.format(N)
df2[label] = np.stack(times100)
df2.to_csv('times/{}_times_20.csv'.format(method.upper()),index=False)
