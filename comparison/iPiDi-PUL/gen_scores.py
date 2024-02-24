import numpy as np
import pandas as pd
import pickle

i = 4

file_name = f"pred_rf_{i}.csv"
csv = pd.read_csv(file_name, index_col=0).values
adj_df = pd.read_csv(r"..\data\adj.csv", index_col=0)
adj_np = adj_df.values

with open(r"../data/fold_info.pickle", "rb") as f:
    fold_info = pickle.load(f)

pos_train_ij_list = fold_info["pos_train_ij_list"]
pos_test_ij_list = fold_info["pos_test_ij_list"]
unlabelled_train_ij_list = fold_info["unlabelled_train_ij_list"]
unlabelled_test_ij_list = fold_info["unlabelled_test_ij_list"]


pos_train_ij = pos_train_ij_list[i]
pos_test_ij = pos_test_ij_list[i]
unlabelled_train_ij = unlabelled_train_ij_list[i]
unlabelled_test_ij = unlabelled_test_ij_list[i]

test_mask_np = np.zeros_like(adj_np)
test_mask_np[tuple(list(pos_test_ij.T))] = 1
test_mask_np[tuple(list(unlabelled_test_ij.T))] = 1

test_idx = np.argwhere(test_mask_np == 1)

scores = csv[tuple(list(test_idx.T))]
np.save(rf"f{i}_e{199}_scores.npy", scores)
