import numpy as np
from numba import jit,njit,typed,types,vectorize
from matplotlib import pyplot as plt
import pandas as pd

from basis import categorical,normalize

#This turns the vector of kinetic parameters and a value of the species number into an array where each element if the rate of that 'reaction'
@njit
def get_rates(x,value):
    birth,d = value
    E = x
    rates = [birth,d*E]
    return np.array(rates)

S = np.array([1,-1]) #Stoichiometry of each reaction.


#This is a Guillespie algorithm. It is a realization of the process starting at x0 and evolving according to the
#kinetic parameters in value up to time T.
def run_forward(x0,T,value):
    t = 0
    x = x0
    xs = [x*1.0]
    ts=[0.0]
    i=0
    while True:
        rates = get_rates(x,value)
        if rates.sum()==0:
            return np.array(ts),np.stack(xs)

        jump_t = np.random.exponential(1/rates.sum())
        
        t += jump_t
        
        if T<t:
            return np.array(ts),np.stack(xs)
               
        reaction = categorical(normalize(rates))
        x+=S[reaction]
        
        xs.append(x+0)
        ts.append(t*1.0)
        
#Run the synthetic data experiment, with T_all being an array of stopping times.
def run_experiment(ground,T_all,x0=0):    
    w_all = []
    for t_f in T_all:
        y = run_forward(x0,t_f,ground)[1][-1]
        w_all.append(y)
        
    return ground,T_all,np.array(w_all)