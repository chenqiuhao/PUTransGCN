import time

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib
import pickle
import os
from sklearn.naive_bayes import GaussianNB

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)

plot_dir = [
    "./PDformer_spy",
    "./PDformer_two_step",
    "./PDformer_pu_bagging",
]
adj_np = pd.read_csv(r".\data\adj.csv", index_col=0).values
p_sim_np = pd.read_csv(r".\data\p2p_smith.csv", index_col=0).values
d_sim_np = pd.read_csv(r".\data\d2d_do.csv", index_col=0).values

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

with open(r"data/fold_info.pickle", "rb") as f:
    fold_info = pickle.load(f)

pos_train_ij_list = fold_info["pos_train_ij_list"]
pos_test_ij_list = fold_info["pos_test_ij_list"]
unlabelled_train_ij_list = fold_info["unlabelled_train_ij_list"]
unlabelled_test_ij_list = fold_info["unlabelled_test_ij_list"]
p_gip_list = fold_info["p_gip_list"]
d_gip_list = fold_info["d_gip_list"]


for dir in plot_dir:
    with open(f"{dir}/rn_ij_list.pickle", "rb") as f:
        rn_ij_list = pickle.load(f)
    for i in range(5):
        print(f"fold {i}")
        pos_train_ij = pos_train_ij_list[i]
        pos_test_ij = pos_test_ij_list[i]
        unlabelled_train_ij = unlabelled_train_ij_list[i]
        unlabelled_test_ij = unlabelled_test_ij_list[i]
        rn_ij = rn_ij_list[i]

        prob_mat = np.zeros_like(adj_np)
        cnt = np.zeros_like(adj_np)

        train_ij = np.vstack((pos_train_ij, rn_ij))
        train_feat = feat_mat[tuple(list(train_ij.T))]
        train_label = adj_np[tuple(list(train_ij.T))]

        pos_test_feat = feat_mat[tuple(list(pos_test_ij.T))]
        # pos_test_label = adj_np[tuple(list(pos_test_ij.T))]

        clf = GaussianNB()
        clf.fit(train_feat, train_label)

        pos_test_prob = clf.predict_proba(pos_test_feat)[:, 1]
        plt.figure()
        plt.title(f"{dir[11:]} fold{i} pos test score")
        counts, edges, bars = plt.hist(pos_test_prob, range=(0, 1))
        plt.bar_label(bars)
        plt.grid()

plt.show()
