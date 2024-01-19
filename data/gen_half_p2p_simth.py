import pandas as pd
import itertools
import numpy as np


def matrix(a, b, match_score=3, gap_cost=2):
    H = np.zeros((len(a) + 1, len(b) + 1), int)

    for i, j in itertools.product(range(1, H.shape[0]), range(1, H.shape[1])):
        match = H[i - 1, j - 1] + (
            match_score if a[i - 1] == b[j - 1] else -match_score
        )
        delete = H[i - 1, j] - gap_cost
        insert = H[i, j - 1] - gap_cost
        H[i, j] = max(match, delete, insert, 0)
    return H


piRNA_seq_csv = pd.read_csv(r"piRNA_seq.csv")
piRNA_seq_dict = dict(piRNA_seq_csv.values)

p2p = pd.DataFrame(columns=list(piRNA_seq_dict.keys()), index=list(piRNA_seq_dict.keys()))
cnt = 0
import time

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
        sim = matrix(index_seq, j_seq, match_score=3, gap_cost=2).max()
        # sim = 1
        len_j_seq = len(j_seq)
        p2p.at[index, j_name] = sim / np.sqrt(len_index * len_j_seq) / 3
b = time.time()
print(b - a)
p2p.to_csv(r"half_p2p_smith.csv")
c = time.time()
print(c - b)
