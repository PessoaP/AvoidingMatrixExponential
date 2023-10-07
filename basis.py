import numpy as np
from scipy.special import gammaln
from numba import njit,typed,types
from scipy.optimize import curve_fit

#these are compiled functions for the forward simulation
@njit
def categorical(p):
    return (p.cumsum()<np.random.rand()).argmin()
@njit
def normalize(x):
    return x/x.sum()

#these are functions for inference, ideally they would be compiled. However, since we lack a numba compatible implementation of log gamma we will use the scipy version of it in an uncompiled form.

def log_factorial(n):
    return gammaln(n+1)

def log_choose(n, k):
    return log_factorial(n) - log_factorial(k) - log_factorial(n - k)

def loglike_binomial(n, k, beta):
    return log_choose(n, k) + k * np.log(beta) + (n - k) * np.log(1 - beta)

def loglike_poisson(k, rate):
    return k * np.log(rate) - rate - log_factorial(k)


#These are functions for the BD inference
def gaussian_likelihood(x, mu, var):
    exponent = (x-mu)**2/(2*var)
    coefficient = 1 / np.sqrt(2*np.pi*var)    
    return coefficient * np.exp(-exponent)

def lprior(th):
    return np.log(gaussian_likelihood(np.log(th),1,1e5)).sum()

def BD_mass(t,b,g):
    return b/g*(1-np.exp(-g*t))

def naive_estimation(T_all,w_all):
    (beta,gamma), pcov_ieu = curve_fit(BD_mass, T_all,w_all)
    return beta,gamma

@njit
def make_initial(center,var,N):
    if var>0:
        exponent = -(np.arange(N)-center)**2/(2*var)
        rho = np.exp(exponent)#*1/(var*np.sqrt(2*np.pi))
        return normalize(rho)
    rho = np.zeros(N)
    rho[center]+=1
    return rho

@njit
def arr_replace(old,new,replace):
    res = np.zeros_like(old)
    res[replace] = new[replace]
    res[np.logical_not(replace)] = old[np.logical_not(replace)]
    return old

