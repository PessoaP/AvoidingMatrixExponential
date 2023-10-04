import numpy as np
from numba import njit,typed,types,prange
from numba.experimental import jitclass

@njit
def sm_asarray(lines,columns,values,shape):
    res = np.zeros((shape,shape))
    for i in range(lines.size):
        res[lines[i],columns[i]] += values[i]
    return res
@njit
def sm_sum(lines,columns,values,shape):
    res = np.zeros(shape)
    ind=lines
    for i in range(lines.size):
        res[ind[i]] += values[i]
    return res

@jitclass([('lines', types.int64[:]),
           ('columns', types.int64[:]),
           ('values', types.float64[:]),
           ('shape',types.int64)])    
class sparse_matrix:
    def __init__ (self,lins,cols,values,shap=0):
        self.lines=lins
        self.columns=cols
        self.values=values
        self.shape=shap
        if shap == 0:
            self.shape=1+lins.max()#assume always square
    
    def __add__(self,other):
        return sparse_matrix(np.concatenate((self.lines,other.lines)),
                             np.concatenate((self.columns,other.columns)),
                             np.concatenate((self.values,other.values)),
                             max(self.shape,other.shape))
            
    def to_dense(self):
        return sm_asarray(self.lines,self.columns,self.values,self.shape)

    def line_sum(self):
        return sm_sum(self.lines,self.columns,self.values,self.shape)
    
@njit
def dot(arr,sm):
    res=np.zeros(sm.shape,types.float64)
    for (line,col,val) in zip(sm.lines,sm.columns,sm.values):
        res[col] += val*arr[line]
    return res

@njit
def sm_times_array(sm,arr):
    res=np.zeros(sm.shape,types.float64)
    for (line,col,val) in zip(sm.lines,sm.columns,sm.values):
        res[line] += val*arr[col]
    return res

@njit
def get_A(sm):
    lins = np.concatenate((sm.lines,np.arange(sm.shape)))
    cols = np.concatenate((sm.columns,np.arange(sm.shape)))
    values = np.concatenate((sm.values,-sm_sum(sm.lines,sm.columns,sm.values,sm.shape)))
    return sparse_matrix(lins,cols,values,sm.shape)
    #return sm + sparse_matrix(np.arange(sm.shape),
    #                          np.arange(sm.shape),
    #                          -sm.line_sum(),
    #                          sm.shape)
    
@njit
def get_B(sm):
    newlines=np.arange(sm.shape)
    b_lines=np.concatenate((newlines,sm.lines))
    b_columns=np.concatenate((newlines,sm.columns))

    #ad = sm.line_sum()
    ad =  sm_sum(sm.lines,sm.columns,sm.values,sm.shape)
    omega = 1.1*ad.max()
    b_values = np.concatenate(((1-ad/omega),(sm.values/omega)))
    return sparse_matrix(b_lines,b_columns,b_values,sm.shape),omega
