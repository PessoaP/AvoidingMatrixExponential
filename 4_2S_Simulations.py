#!/usr/bin/env python
# coding: utf-8

# In[1]:


from forward_2S import *
import pandas as pd


N_r = 300
beta = 1000

x = np.random.normal(size=10)

# In[2]:


gamma = 1.0
l01 = 2
l10 = 1

N_DNA = 2

T_all = np.arange(.5,10.1,.5)
T_all = np.sort(np.array(N_r*list(T_all)))


# In[3]:


ground = np.array((beta,gamma,l01,l10))
ground, T_all, W_all = run_experiment(ground,T_all,x0=0)


# In[4]:


df = pd.DataFrame()

df['times'] = np.round(T_all,5)
df['counts'] = W_all

df.to_csv('synthetic_data/2S_synthetic_data--beta={}.csv'.format(beta),index=False)

