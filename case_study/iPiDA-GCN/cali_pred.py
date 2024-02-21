import pandas as pd
import numpy as np

pred_np = pd.read_csv('pred_np_raw.csv', header=None).values
max = pred_np.max().max()
pred_np = pred_np/max
np.savetxt('pred_np.csv', pred_np, delimiter=",")

a=1
