import IEU_inverse as ieu

import pandas as pd
import time

import numpy as np 
from matplotlib import pyplot as plt

import sys

N_sam = 100*20
#np.random.seed(12)


beta_gt = int(sys.argv[1])
gamma_gt = 1

df = pd.read_csv('synthetic_data/synthetic_data--beta={}.csv'.format(beta_gt))
w_all = (df['counts']).to_numpy(int)
T_all = (df['times']).to_numpy()
N = 2*w_all.max()

S_prop = np.eye(2)*1e-8

ground = np.array((beta_gt,gamma_gt))
th_gt = ieu.params(ground,N)

ll_gt_list = []
k_all = 1+(T_all*th_gt.omega).astype(int)
llw = th_gt.loglike_w_k(w_all,k_all)
llk = th_gt.loglike_k(k_all,T_all)
for i in range(10):
    llw,llk,k_all = ieu.update_k(llw,llk,k_all,w_all,T_all,th_gt)
    ll_gt_list.append(llw.sum()+llk.sum())

ll_gt = np.mean(ll_gt_list)
ll_gt

beta,gamma = ieu.naive_estimation(T_all,w_all)

theta = np.array((beta,gamma))#.to(ieu.device)
th_ieu = ieu.params(theta,N)

th = th_ieu

k_all = th_ieu.sample_k(T_all)+w_all
k_all = 1+(T_all*th.omega).astype(int)
llw = th_ieu.loglike_w_k(w_all,k_all)
llk = th_ieu.loglike_k(k_all,T_all)

llw_list =[]
llk_list =[]
th_list =[]

times100 =[]

start=time.time()
for i in range(len(llw_list),N_sam+1):
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
        print(end-start)
        start=time.time()   

llw_pd = np.stack(llw_list)
llk_pd = np.stack(llk_list)
th_pd = np.stack(th_list)

df = pd.DataFrame(th_pd,columns=['birth rate','death rate'])
df['log p(w|k,th)'] = llw_pd[-len(th_pd):]
df['log p(k|th)'] = llk_pd[-len(th_pd):]

#df.to_csv('inference/IEU_inference_beta={}_20.csv'.format(beta_gt),index=False)

try:
    df2 = pd.read_csv('times/IEU_times_20.csv')
except:
    df2 = pd.DataFrame()
label = 'N={}'.format(N)
df2[label] = np.stack(times100)
df2.to_csv('times/IEU_times_20.csv',index=False)




