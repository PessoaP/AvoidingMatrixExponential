import RK_inverse_STS as rk
import KRY_inverse_STS as kry
import IMU_inverse_STS as imu

import pandas as pd
import time

import numpy as np 
from matplotlib import pyplot as plt

import sys

method = sys.argv[3]
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


beta_R_gt, beta_P_gt = float(sys.argv[1]),float(sys.argv[2])

l01_gt = .1
l10_gt = .05

gamma_R_gt = 1.
gamma_P_gt = .1


df = pd.read_csv('synthetic_data/STS_synthetic_data--beta={}-{}.csv'.format(beta_R_gt,beta_P_gt))

w_all = df[['counts_P','counts_R']].to_numpy().astype(int)
T_all = df['times'].to_numpy()
NP,NR = 2*w_all.max(axis=0)

S_prop = np.eye(6)*1e-8


ground = np.array((beta_R_gt,beta_P_gt,l01_gt,l10_gt,gamma_R_gt,gamma_P_gt))
th_gt = met.params(ground,NP,NR)

ll_gt = th_gt.loglike_w(w_all,T_all)
ll_gt.sum()

theta = (.5+np.random.rand(6))*ground
th = met.params(theta,NP,NR)

ll = th.loglike_w(w_all,T_all)

met.update_th(ll,w_all,T_all,th,S_prop)

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
        


# In[11]:


#Restarting adaptative
theta = th_list[np.argmax(ll_list)]
th = met.params(theta,NP,NR)
S_prop = np.eye(6)*1e-8

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


# In[12]:


#Restarting again

ll_list =[]
th_list =[]

times100 =[]


# In[13]:


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
                met.save(ll_list,th_list,beta_R_gt,beta_P_gt)
        
        print(end-start)            
        start=time.time()   



# In[ ]:


met.save(ll_list,th_list,beta_R_gt,beta_P_gt)


# In[ ]:




