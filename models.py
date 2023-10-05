from basis import *
import smn

@njit
def lam_1S(birth,d,N):    
    states = np.arange(N)
    
    above_l = np.arange(N-1)
    above_c = np.arange(1,N)
    above_v = birth + 0.0*above_l
    #above = smn.sparse_matrix(above_l,above_c,above_v,N)
    
    below_l = np.arange(1,N)
    below_c = np.arange(N-1)
    below_v = d*below_l 
    #below = smn.sparse_matrix(below_l,below_c,below_v,N)
    
    #return above+below

    lines = np.concatenate((above_l,below_l))
    cols  = np.concatenate((above_c,below_c))
    vals = np.concatenate((above_v,below_v))
    return smn.sparse_matrix(lines,cols,vals,N)

@njit
def lam_2S(birth,d,l01,l10,N_RNA,N_DNA=2): 
    #in future updates we might give support for other values, now N_DNA has to be 2
    states = np.arange(N_RNA)
    
    above_l = np.arange(N_RNA-1)
    above_c = np.arange(1,N_RNA)    
    above_v_active = birth + 0.0*above_l
    above_active = smn.sparse_matrix(N_RNA+above_l,N_RNA+above_c,above_v_active,N_RNA*N_DNA)

    below_l = np.arange(1,N_RNA)
    below_c = np.arange(N_RNA-1)
    below_v = d*below_l #+ ((mu_E-d)/k)*torch.pow(states,2)
    below_inactive = smn.sparse_matrix(below_l,below_c,below_v,N_RNA*N_DNA)
    below_active = smn.sparse_matrix(N_RNA+below_l,N_RNA+below_c,below_v,N_RNA*N_DNA)
    
    activation = smn.sparse_matrix(states,N_RNA+states,l01+0*states,N_RNA*N_DNA)
    deactivation = smn.sparse_matrix(states+N_RNA,states,l10+0*states,N_RNA*N_DNA)
    
    #return above_active+below_active+below_inactive+activation+deactivation

    lines = np.concatenate((above_active.lines,below_active.lines,below_inactive.lines,activation.lines,deactivation.lines))
    cols  = np.concatenate((above_active.columns,below_active.columns,below_inactive.columns,activation.columns,deactivation.columns))
    vals = np.concatenate((above_active.values,below_active.values,below_inactive.values,activation.values,deactivation.values))
    return smn.sparse_matrix(lines,cols,vals,N_RNA*N_DNA)

@njit
def lam_STS(beta_R, beta_P, l01, l10, gamma_R, gamma_P,NP,NR): 
    init = np.arange(2*NR*NP)

    nR = (init%NR).astype(np.int64)
    nP = ((init/NR)%NP).astype(np.int64)
    s = ((init/(NP*NR))%2).astype(np.int64)

    #first reaction -- produce R

    keep = np.logical_and(s==0,nR!=NR-1)
    pal = init[keep]
    val = (beta_R)*np.ones_like(pal)
    pac = pal+1

    prod_R = smn.sparse_matrix(pal,pac,val,2*NP*NR)

    #second reaction -- produce P

    keep = np.logical_and(nP!=NP-1,nR!=0)
    pal = init[keep]
    val = nR[keep]*beta_P
    pac = pal+NR

    prod_P = smn.sparse_matrix(pal,pac,val,2*NP*NR)

    #third reaction -- deactivate

    keep = np.logical_and(s==0,nP!=0)
    pal = init[keep]
    val = nP[keep]*l01
    pac = pal+NP*NR-NR

    deact = smn.sparse_matrix(pal,pac,val,2*NP*NR)

    #fourth reaction -- activate

    keep = np.logical_and(s==1,nP!=NP-1)
    pal = init[keep]
    val = np.ones_like(pal)*l10
    pac = pal-NP*NR+NR

    acti = smn.sparse_matrix(pal,pac,val,2*NP*NR)

    #fifth reaction -- degrade R
    keep = nR!=0
    pal = init[keep]
    val = gamma_R*nR[keep]
    pac = pal-1

    deg_R = smn.sparse_matrix(pal,pac,val,2*NP*NR)

    #sixth reaction -- degrade P
    keep = nP!=0
    pal = init[keep]
    val = gamma_P*nP[keep]
    pac = pal-NR

    deg_P = smn.sparse_matrix(pal,pac,val,2*NP*NR)

    #return prod_R + prod_P + deact + acti + deg_R + deg_P
    lines = np.concatenate((prod_R.lines,prod_P.lines,deact.lines,acti.lines,deg_R.lines,deg_P.lines))
    cols  = np.concatenate((prod_R.columns,prod_P.columns,deact.columns,acti.columns,deg_R.columns,deg_P.columns))
    vals = np.concatenate((prod_R.values,prod_P.values,deact.values,acti.values,deg_R.values,deg_P.values))
    return smn.sparse_matrix(lines,cols,vals,NR*NP*2)