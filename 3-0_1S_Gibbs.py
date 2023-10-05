import RK_inverse as rk
import KRY_inverse as kry
import IMU_inverse as imu
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
    'imu': imu,
    'me': me
}
met = methods.get(method)
if not met:
    print('method not recognized')


N_sam = 100*2500
np.random.seed(10)

beta_gt = int(sys.argv[1])
gamma_gt = 1



df = pd.read_csv('synthetic_data/synthetic_data--beta={}.csv'.format(beta_gt))
w_all = (df['counts']).to_numpy(int)
T_all = (df['times']).to_numpy()
N = 2*w_all.max()

S_prop = np.eye(2)*1e-8

ground = np.array((beta_gt,gamma_gt))
th_gt = met.params(ground,N)

ll_gt = th_gt.loglike_w(w_all,T_all) 
#this guarantees everything that can be compiled is compiled

beta,gamma = met.naive_estimation(T_all,w_all)

theta = np.array((beta,gamma))
th = met.params(theta,N)
ll = th.loglike_w(w_all,T_all)

met.update_th(ll,w_all,T_all,th,S_prop)


# **Doing adaptive**

ll_list =[]
th_list =[]

mat_list=[]
mat_list.append(S_prop)

acc_count=0
i = 0

start=time.time() 
while acc_count<10:
    i+=1
    ll,th  = met.update_th(ll,w_all,T_all,th,S_prop)    
    ll_list.append(ll.sum())
    th_list.append(th.value)
    
    if i%100 == 0:
        end=time.time()
        ll_last = np.stack(ll_list[-101:])
        th_last  = np.stack(th_list[-101:])

        print(ll_last.mean())
        print(th_last.mean(axis=0))
        if i>0:
            print('iteration ',i)
            #times100.append(end-start)
            acceptance = np.mean(((th_last[1:]-th_last[:-1]).mean(axis=1)!=0))
            if acceptance>.2 and acceptance<.5:
                acc_count+=1
            else:
                acc_count = 0
            print('accept rate',acceptance)
            if i%300 == 0:
                S_prop = met.update_S(th_last)
                mat_list.append(S_prop)
                print(S_prop)
            print(end-start)
        start=time.time() 


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
            if i%2000 == 0:
                met.save(ll_list,th_list,beta_gt)
        
        print(end-start)            
        start=time.time()   

met.save(ll_list,th_list,beta_gt)
