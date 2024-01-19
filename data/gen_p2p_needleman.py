import pandas as pd
import numpy as np


half_p2p_needleman = pd.read_csv(r"half_p2p_needleman.csv", index_col=0)
half_p2p_needleman_values = half_p2p_needleman.values
half_p2p_needleman_values[np.isnan(half_p2p_needleman_values)] = 0
half_p2p_needleman_values = half_p2p_needleman_values + half_p2p_needleman_values.T
# half_p2p_needleman_values[half_p2p_needleman_values < 0] = 0
max = half_p2p_needleman_values.max()
min = half_p2p_needleman_values.min()
half_p2p_needleman_values = (half_p2p_needleman_values - min) / (max - min)
np.fill_diagonal(half_p2p_needleman_values, 1)
index = half_p2p_needleman.index
p2p_needleman = pd.DataFrame(half_p2p_needleman_values, columns=index, index=index)
p2p_needleman.to_csv(r"p2p_needleman.csv")
