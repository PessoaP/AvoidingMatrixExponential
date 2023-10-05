import RK_inverse_2S as rk
import KRY_inverse_2S as kry
import IMU_inverse_2S as imu

import pandas as pd
import time

import numpy as np 
from matplotlib import pyplot as plt

import sys

method = sys.argv[2]
methods = {
    'rk': rk,
    'kry': kry,
    'imu': imu,
}
met = methods.get(method)
if not met:
    print('method not recognized')


N_sam = 100*20
np.random.seed(10000)

beta_gt = int(sys.argv[1])
gamma_gt = 1.0
l01_gt = 2
l10_gt = 1

df = pd.read_csv('synthetic_data/2S_synthetic_data--beta={}.csv'.format(beta_gt))
w_all = (df['counts']).to_numpy(np.int64)
T_all = (df['times']).to_numpy()
N_RNA = 2*w_all.max()

S_prop = np.eye(4)*1e-8

ground = np.array((beta_gt,gamma_gt,l01_gt,l10_gt))
th_gt = met.params(ground,N_RNA)

ll_gt_list = []
ll = th_gt.loglike_w(w_all,T_all)

#init at gound truth
theta = 1.0*ground
th = met.params(theta,N_RNA)

# **Doing adaptative**

ll_list =[]
th_list =[]

times100 =[]

start=time.time()
for i in range(len(ll_list),N_sam):
    ll,th  = met.update_th(ll,w_all,T_all,th,S_prop)
    
    ll_list.append(ll)
    th_list.append(th.value)
    
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
    df2 = pd.read_csv('times/2S_{}_times_20.csv'.format(method.upper()))
except:
    df2 = pd.DataFrame()
label = 'N={}'.format(th.N)
df2[label] = np.stack(times100)
df2.to_csv('times/2S_{}_times_20.csv'.format(method.upper()),index=False)
