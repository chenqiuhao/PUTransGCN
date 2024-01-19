import pandas as pd
import itertools
import numpy as np
from minineedle import needle, smith, core
import time


piRNA_seq_csv = pd.read_csv(r"piRNA_seq.csv")
piRNA_seq_dict = dict(piRNA_seq_csv.values)


p2p = pd.DataFrame(index=list(piRNA_seq_dict.keys()), columns=list(piRNA_seq_dict.keys()))
cnt = 0

a = time.time()
for index, row in p2p.iterrows():
    # if cnt > 10:
    #     break
    print(cnt)
    cnt += 1
    row_index = list(row.index)
    row_index2 = row_index[cnt:]
    index_seq = piRNA_seq_dict[index]
    len_index = len(index_seq)
    for j_name in row_index2:
        j_seq = piRNA_seq_dict[j_name]
        alignment = needle.NeedlemanWunsch(j_seq, index_seq)
        sim = alignment.get_score()
        p2p.at[index, j_name] = sim
b = time.time()
print(b - a)
p2p.to_csv(r"half_p2p_needleman.csv")
c = time.time()
print(c - b)
