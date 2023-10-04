from forward import *
import pandas as pd
np.random.seed(10)
import sys

N_r = 300
beta = int(sys.argv[1])

gamma=1

T_all = np.arange(.5,5.1,.5)
T_all = np.sort(np.array(N_r*list(T_all)))

ground, T_all, W_all = run_experiment((beta,gamma),T_all,x0=0)

df = pd.DataFrame()

df['times'] = np.round(T_all,5)
df['counts'] = W_all

df.to_csv('synthetic_data/synthetic_data--beta={}.csv'.format(beta),index=False)
