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
path = "classifier/"
if not os.path.exists(path):
    os.makedirs(path)


rn_ij_list = []
for i in range(5):
    print(f"fold {i}")
    pos_train_ij = pos_train_ij_list[i]
    pos_test_ij = pos_test_ij_list[i]
    unlabelled_train_ij = unlabelled_train_ij_list[i]
    unlabelled_test_ij = unlabelled_test_ij_list[i]

    train_ij = np.vstack((pos_train_ij, unlabelled_train_ij))
    train_feat = feat_mat[tuple(list(train_ij.T))]
    train_label = adj_np[tuple(list(train_ij.T))]
    ys = (2 * train_label - 1).copy()
    regressor1 = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    regressor1.fit(train_feat, train_label)

    with open(rf".\classifier\rfclf_f85_e500_{i}.pickle", "wb") as f:
        pickle.dump(regressor1, f)

    #with open(rf".\classifier\rfclf_f85_e500_{i}.pickle", "rb") as f:
    #    regressor1 = pickle.load(f)
    train_prob = regressor1.predict_proba(train_feat)[:, 1]

    range_pos = [min(train_prob * (ys > 0)), max(train_prob * (ys > 0))]

    p_new = ys[(ys < 0) & (train_prob >= range_pos[1])]
    n_new = ys[(ys < 0) & (train_prob <= range_pos[0])]
    ys[(ys < 0) & (train_prob >= range_pos[1])] = 1
    ys[(ys < 0) & (train_prob <= range_pos[0])] = 0

    regressor2 = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    for j in range(5):
        print("Iteration: ", j, "len(p_new)", len(p_new), "len(n_new)", len(n_new))
        if len(p_new) + len(n_new) == 0 and i > 0:
            break

        a = time.time()
        regressor2.fit(train_feat, ys)
        b = time.time()
        print(b - a)

        train_prob = regressor2.predict_proba(train_feat)[:, -1]

        # Find the range of scores given to positive data points
        range_pos = [min(train_prob * (ys > 0)), max(train_prob * (ys > 0))]

        # Repeat step 1
        p_new = ys[(ys < 0) & (train_prob >= range_pos[1])]
        n_new = ys[(ys < 0) & (train_prob <= range_pos[0])]
        ys[(ys < 0) & (train_prob >= range_pos[1])] = 1
        ys[(ys < 0) & (train_prob <= range_pos[0])] = 0

    rn_ij = train_ij[ys == 0]
    rn_ij_list.append(rn_ij)

with open("rn_ij_list.pickle", "wb") as f:
    pickle.dump(rn_ij_list, f)
