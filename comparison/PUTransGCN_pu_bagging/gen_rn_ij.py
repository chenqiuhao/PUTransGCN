import time

import numpy as np
from utils import *
import torch
from model import *
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib
import pickle
import os

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

seed_everything(42)

adj_np = pd.read_csv(r"..\data\adj.csv", index_col=0).values
p_sim_np = pd.read_csv(r"..\data\p2p_smith.csv", index_col=0).values
d_sim_np = pd.read_csv(r"..\data\d2d_do.csv", index_col=0).values

num_p, num_d = adj_np.shape
n_pca_p_sim = 85
pca_p_sim = PCA(n_components=n_pca_p_sim)
pca_p_sim_feat = pca_p_sim.fit_transform(p_sim_np)
pca_d_sim = PCA()
pca_d_sim_feat = pca_d_sim.fit_transform(d_sim_np)

feat_mat = np.zeros((num_p, num_d, (n_pca_p_sim + num_d)))

for i in range(num_p):
    for j in range(num_d):
        feat_mat[i, j] = np.append(pca_p_sim_feat[i], pca_d_sim_feat[j])


with open(r"../data/fold_info.pickle", "rb") as f:
    fold_info = pickle.load(f)

pos_train_ij_list = fold_info["pos_train_ij_list"]
pos_test_ij_list = fold_info["pos_test_ij_list"]
unlabelled_train_ij_list = fold_info["unlabelled_train_ij_list"]
unlabelled_test_ij_list = fold_info["unlabelled_test_ij_list"]
p_gip_list = fold_info["p_gip_list"]
d_gip_list = fold_info["d_gip_list"]


T = 10
rn_ij_list = []
for i in range(5):
    print(f"fold {i}")
    pos_train_ij = pos_train_ij_list[i]
    pos_test_ij = pos_test_ij_list[i]
    unlabelled_train_ij = unlabelled_train_ij_list[i]
    unlabelled_test_ij = unlabelled_test_ij_list[i]

    prob_mat = np.zeros_like(adj_np)
    cnt = np.zeros_like(adj_np)
    for t in range(T):
        print(t)
        unlabelled_train_ij_t_idx = np.random.choice(
            np.arange(len(unlabelled_train_ij)), replace=False, size=len(pos_train_ij)
        )
        unlabelled_train_ij_t = unlabelled_train_ij[unlabelled_train_ij_t_idx]
        rest_unlabelled_train_ij_t_idx = np.setdiff1d(
            np.arange(len(unlabelled_train_ij)), unlabelled_train_ij_t_idx
        )
        rest_unlabelled_train_ij_t = unlabelled_train_ij[rest_unlabelled_train_ij_t_idx]

        train_ij = np.vstack((pos_train_ij, unlabelled_train_ij_t))
        train_feat = feat_mat[tuple(list(train_ij.T))]
        train_label = adj_np[tuple(list(train_ij.T))]
        rest_unlabelled_train_feat = feat_mat[tuple(list(rest_unlabelled_train_ij_t.T))]

        regressor = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)

        regressor.fit(train_feat, train_label)

        rest_unlabelled_train_prob = regressor.predict_proba(
            rest_unlabelled_train_feat
        )[:, 1]
        prob_mat[
            tuple(list(rest_unlabelled_train_ij_t.T))
        ] += rest_unlabelled_train_prob
        cnt[tuple(list(rest_unlabelled_train_ij_t.T))] += 1

    fnl_score = prob_mat / cnt
    np.savetxt(f"fnl_score_{i}.csv", fnl_score, delimiter=",")

    non_nan_indexes = np.nonzero(~np.isnan(fnl_score))
    non_nan_values = fnl_score[non_nan_indexes]
    non_nan_ij = np.transpose(non_nan_indexes)

    pos_prob = regressor.predict_proba(feat_mat[tuple(list(pos_train_ij.T))])[:, 1]
    pos_prob_min = pos_prob.min()

    sorted_nums = sorted(enumerate(non_nan_values), key=lambda x: x[1])
    neg_nums = list(filter(lambda x: x[1] < pos_prob_min, sorted_nums))
    idx = [i[0] for i in neg_nums]

    rn_idx = idx[int(len(idx) / 3) : int(len(idx) / 3 * 2)]
    rn_ij = non_nan_ij[rn_idx]
    rn_ij_list.append(rn_ij)


with open("rn_ij_list.pickle", "wb") as f:
    pickle.dump(rn_ij_list, f)
