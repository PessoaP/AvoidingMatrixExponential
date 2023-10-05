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


N_sam = 100*2500
np.random.seed(100)

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

ll_gt = th_gt.loglike_w(w_all,T_all)
ll_gt.sum()




#theta = 1.0*ground
theta = np.concatenate(([w_all.max()],np.ones(3)*1.5))
th = met.params(theta,N_RNA)

ll = th.loglike_w(w_all,T_all)

met.update_th(ll,w_all,T_all,th,S_prop)


# **Doing adaptative**

ll_list =[]
th_list =[]

times100 =[]

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
        
#Restarting adaptative
theta = th_list[np.argmax(ll_list)]
th = met.params(theta,N_RNA)
S_prop = np.eye(4)*1e-8

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

#Restarting again, now for good
ll_list =[]
th_list =[]

times100 =[]


# In[ ]:


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




