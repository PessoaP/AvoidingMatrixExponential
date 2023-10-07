import IEU_inverse_2S as ieu

import pandas as pd
import time

import numpy as np 
from matplotlib import pyplot as plt
import sys

N_sam = 100*2500
np.random.seed(10)

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
th_gt = ieu.params(ground,N_RNA)

ll_gt_list = []
k_all = 1+(T_all*th_gt.omega).astype(int)
llw = th_gt.loglike_w_k(w_all,k_all)
llk = th_gt.loglike_k(k_all,T_all)
for i in range(10):
    llw,llk,k_all = ieu.update_k(llw,llk,k_all,w_all,T_all,th_gt)
    ll_gt_list.append(llw.sum()+llk.sum())

ll_gt = np.mean(ll_gt_list)

theta = 1.0*ground
th_ieu = ieu.params(theta,N_RNA)


k_all = th_ieu.sample_k(T_all)+w_all
k_all = 1+(T_all*th_ieu.omega).astype(int)
llw = th_ieu.loglike_w_k(w_all,k_all)
llk = th_ieu.loglike_k(k_all,T_all)

th=th_ieu

# **Doing adaptative**

llw_list =[]
llk_list =[]
th_list =[]

times100 =[]


mat_list=[]
mat_list.append(S_prop)


acc_count=0
i = 0

start=time.time() 
while acc_count<10:
    i+=1
    llw,llk,th  = ieu.update_th(llw,llk,k_all,w_all,T_all,th,S_prop)
    llw,llk,k_all = ieu.update_k(llw,llk,k_all,w_all,T_all,th)
    
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
            if acceptance>.2 and acceptance<.5:
                acc_count+=1
            else:
                acc_count = 0
            print('accept rate',acceptance)
            if i%300 == 0:
                S_prop = ieu.update_S(th_last)
                mat_list.append(S_prop)
                print(S_prop)
            print(end-start)
        start=time.time() 
        
#Restarting adaptative
theta = th_list[np.argmax(np.array(llw_list)+np.array(llw_list))]
th = ieu.params(theta,N_RNA)
S_prop = np.eye(4)*1e-8

acc_count=0
i = 0

start=time.time() 
while acc_count<10:
    i+=1
    llw,llk,th  = ieu.update_th(llw,llk,k_all,w_all,T_all,th,S_prop)
    llw,llk,k_all = ieu.update_k(llw,llk,k_all,w_all,T_all,th)
    
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
            if acceptance>.2 and acceptance<.5:
                acc_count+=1
            else:
                acc_count = 0
            print('accept rate',acceptance)
            if i%300 == 0:
                S_prop = ieu.update_S(th_last)
                mat_list.append(S_prop)
                print(S_prop)
            print(end-start)
        start=time.time() 


llw_list =[]
llk_list =[]
th_list =[]


start=time.time()
for i in range(len(llw_list),N_sam):
    llw,llk,th  = ieu.update_th(llw,llk,k_all,w_all,T_all,th,S_prop)
    llw,llk,k_all = ieu.update_k(llw,llk,k_all,w_all,T_all,th)
    
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
                ieu.save(llw_list,llk_list,th_list,beta_gt)
        
        print(end-start)            
        start=time.time()   

ieu.save(llw_list,llk_list,th_list,beta_gt)





