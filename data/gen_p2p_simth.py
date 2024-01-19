import pandas as pd
import numpy as np


half_p2p_smith = pd.read_csv(r"half_p2p_smith.csv", index_col=0)
half_p2p_smith_values = half_p2p_smith.values
half_p2p_smith_values[np.isnan(half_p2p_smith_values)] = 0
half_p2p_smith_values = half_p2p_smith_values + half_p2p_smith_values.T
np.fill_diagonal(half_p2p_smith_values, 1)
index = half_p2p_smith.index
p2p_smith = pd.DataFrame(half_p2p_smith_values, columns=index, index=index)
p2p_smith.to_csv(r"p2p_smith.csv")
