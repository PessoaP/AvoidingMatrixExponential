import numpy as np
from numba import jit,njit,typed,types,vectorize
from matplotlib import pyplot as plt
import pandas as pd

from basis import categorical,normalize

#This turns the vector of kinetic parameters and a value of the species number into an array where each element if the rate of that 'reaction'
@njit
def get_rates(x,value,N_RNA,N_DNA):
    birth,d,l01,l10 = value
    E = x%N_RNA
    rates = [birth*(x>=N_RNA),d*E,l01*(x<N_RNA),l10*(x>=N_RNA)]
    targets = [x+1,x-1,x+N_RNA,x-N_RNA]
    return np.array(rates),np.array(targets)



#This is a Guillespie algorithm. It is a realization of the process starting at x0 and evolving according to the
#kinetic parameters in value up to time T.
def run_forward(x0,T,value,N_RNA,N_DNA):
    t = 0
    x = x0
    xs = [x*1.0]
    ts=[0.0]
    i=0
    while True:
        rates,targets = get_rates(x,value,N_RNA,N_DNA)
        if rates.sum()==0:
            return np.array(ts),np.stack(xs)

        jump_t = np.random.exponential(1/rates.sum())
        
        t += jump_t
        
        if T<t:
            return np.array(ts),np.stack(xs)
               
        reaction = categorical(normalize(rates))
        x=targets[reaction]
        
        xs.append(x%N_RNA)
        ts.append(t*1.0)
        
#Run the synthetic data experiment, with T_all being an array of stopping times.
def run_experiment(ground,T_all,N_RNA=5000,N_DNA=2,x0=0):   #so that the simulator does not implement the cutoff, N_RNA is going to be considerably bigger than all cases we will take into account. 
    w_all = []
    for t_f in T_all:
        y = run_forward(x0,t_f,ground,N_RNA,N_DNA)[1][-1]
        w_all.append(y)
        
    return ground,T_all,np.array(w_all)