#!/usr/bin/env python
# coding: utf-8

# In[1]:


import IMU_inverse as imu

import pandas as pd
import time

import numpy as np 
from matplotlib import pyplot as plt

N_sam = 100*1000
np.random.seed(10000)


# In[2]:


beta_gt = 50
gamma_gt = 1

df = pd.read_csv('synthetic_data/synthetic_data--beta={}.csv'.format(beta_gt))
w_all = (df['counts']).to_numpy(int)
T_all = (df['times']).to_numpy()
N = 2*w_all.max()


# In[3]:


S_prop = np.eye(2)*1e-8


# In[4]:


ground = np.array((beta_gt,gamma_gt))
th_gt = imu.params(ground,N)

ll_gt = th_gt.loglike_w(w_all,T_all)
ll_gt.sum()


# In[5]:


beta,gamma = imu.naive_estimation(T_all,w_all)


# In[6]:


theta = np.array((beta,gamma))
th_rk = imu.params(theta,N)

th = th_rk
ll = th.loglike_w(w_all,T_all)

imu.update_th(ll,w_all,T_all,th,S_prop)


# **Doing adaptative**

# In[7]:


ll_list =[]
th_list =[]

#times100 =[]


# In[8]:


mat_list=[]
mat_list.append(S_prop)


# In[9]:


acc_count=0
i = 0

start=time.time() 
while acc_count<10:
    i+=1
    ll,th  = imu.update_th(ll,w_all,T_all,th,S_prop)    
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
                S_prop = imu.update_S(th_last)
                mat_list.append(S_prop)
                print(S_prop)
            print(end-start)
        start=time.time() 
        

    


# **Restarting chain**

# In[10]:


ll_list =[]
th_list =[]

times100 =[]


# In[11]:


start=time.time()
for i in range(len(ll_list),N_sam):
    ll,th  = imu.update_th(ll,w_all,T_all,th,S_prop)    
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
                imu.save_imu(ll_list,th_list,beta_gt)
            
        print(end-start)
        start=time.time()   


# In[13]:


imu.save_imu(ll_list,th_list,beta_gt)


# In[ ]:




