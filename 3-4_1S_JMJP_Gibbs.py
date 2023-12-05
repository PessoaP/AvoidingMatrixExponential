import JMJP_inverse as jmjp

import pandas as pd
import time

import numpy as np 
from matplotlib import pyplot as plt

import sys

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
th_gt = jmjp.params(ground,N)

ll_gt_list = []
k_all = 1+(T_all*th_gt.omega).astype(int)
llw = th_gt.loglike_w_k(w_all,k_all)
llk = th_gt.loglike_k(k_all,T_all)
for i in range(10):
    llw,llk,k_all = jmjp.update_k(llw,llk,k_all,w_all,T_all,th_gt)
    ll_gt_list.append(llw.sum()+llk.sum())

ll_gt = np.mean(ll_gt_list)
#this guarantees everything that can be compiled is compiled

beta,gamma = jmjp.naive_estimation(T_all,w_all)

theta = np.array((beta,gamma))#.to(jmjp.device)
th_jmjp = jmjp.params(theta,N)

th = th_jmjp

k_all = th_jmjp.sample_k(T_all)+w_all
k_all = 1+(T_all*th.omega).astype(int)
llw = th_jmjp.loglike_w_k(w_all,k_all)
llk = th_jmjp.loglike_k(k_all,T_all)

# **Doing adaptive**

llw_list =[]
llk_list =[]
th_list =[]

mat_list=[]
mat_list.append(S_prop)

acc_count=0
i = 0

start=time.time() 
while acc_count<10:
    i+=1
    llw,llk,th  = jmjp.update_th(llw,llk,k_all,w_all,T_all,th,S_prop)
    llw,llk,k_all = jmjp.update_k(llw,llk,k_all,w_all,T_all,th)
    
    llw_list.append(llw.sum())
    llk_list.append(llk.sum())
    th_list.append(th.value)
    
    if i%100 == 0:
        end=time.time()
        llw_last = np.stack(llw_list[-101:])
        llk_last = np.stack(llk_list[-101:])
        th_last  = np.stack(th_list[-101:])

        print(llw_last.mean()+llk_last.mean())
        print(th_last.mean(axis=0))
        if i>0:
            print('iteration ',i)
            #times100.append(end-start)
            acceptance = np.mean(((th_last[1:]-th_last[:-1]).mean(axis=1)!=0))
            if acceptance>=.2 and acceptance<=.5:
                acc_count+=1
            else:
                acc_count = 0
            print('accept rate',acceptance)
            if i%300 == 0:
                S_prop = jmjp.update_S(th_last)
                mat_list.append(S_prop)
                print(S_prop)
            print(end-start)
        start=time.time() 
        


# **restarting chain**

llw_list =[]
llk_list =[]
th_list =[]

times100 =[]

start=time.time()
for i in range(len(llw_list),N_sam):
    llw,llk,th  = jmjp.update_th(llw,llk,k_all,w_all,T_all,th,S_prop)
    llw,llk,k_all = jmjp.update_k(llw,llk,k_all,w_all,T_all,th)
    
    llw_list.append(llw.sum())
    llk_list.append(llk.sum())
    th_list.append(th.value)
    
    if i%100 == 0:
        end=time.time()
        llw_last = np.stack(llw_list[-101:])
        llk_last = np.stack(llk_list[-101:])
        th_last  = np.stack(th_list[-101:])

        print(llw_last.mean()+llk_last.mean())
        print(th_last.mean(axis=0))
        
        if i>0:
            print('iteration ',i)
            times100.append(end-start)
            print('accept rate',np.mean(((th_last[1:]-th_last[:-1]).mean(axis=1)!=0)))
            if i%2000 == 0:
                jmjp.save(llw_list,llk_list,th_list,beta_gt)
        print(end-start)
        start=time.time()   

jmjp.save(llw_list,llk_list,th_list,beta_gt)

