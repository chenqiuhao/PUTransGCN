import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
p_sim_pred = np.loadtxt('p_sim_pred.csv',delimiter=",")
p2p_smith = pd.read_csv('../data/p2p_smith.csv', index_col=0).values
# p2p_smith = pd.read_csv('../data/p2p_needleman.csv', index_col=0).values
with open(r"../data/p_gip.pickle", "rb") as f:
    p_gip = pickle.load(f)
pcc_list = []
for i in range(p_sim_pred.shape[0]):
    p_sim_pred_i = p_sim_pred[i]
    p2p_smith_i = p2p_smith[i]
    # p2p_smith_i2 = p_gip[i]
    # p2p_smith_i = (p2p_smith_i1+p2p_smith_i2)/2
    pc = np.corrcoef(p_sim_pred_i, p2p_smith_i)
    pcc_list.append(pc[0,1])

ref = np.zeros_like(pcc_list)
plt.hist(pcc_list)
plt.show(block=True)
stats.ttest_ind(pcc_list, ref, equal_var=1)
a=1