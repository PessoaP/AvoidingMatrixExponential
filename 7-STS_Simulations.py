from forward_STS import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

np.random.seed(1)

N_r = 300

beta_R = 10.*4
beta_P = .5*4

l01 = .1
l10 = .05

gamma_R = 1.
gamma_P = .1

T_all = (np.arange(.5,10,.5))
T_all = np.sort(np.array(N_r*list(T_all)))

ground = np.array((beta_R,beta_P,l01,l10,gamma_R,gamma_P))
ground, T_all, W_all = run_experiment(ground,T_all)


df = pd.DataFrame()

df['times'] = np.round(T_all,5)
df['counts_P'] = W_all[:,0]
df['counts_R'] = W_all[:,1]

df.to_csv('synthetic_data/STS_synthetic_data--beta={}-{}.csv'.format(beta_R,beta_P),index=False)

