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
piRNA_seq_csv = pd.read_csv(r"..\data\piRNA_seq.csv")
piRNA_seq_dict = dict(piRNA_seq_csv.values)

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
    p_gip = p_gip_list[i]
    d_gip = d_gip_list[i]

    train_ij = np.vstack((pos_train_ij, unlabelled_train_ij))
    train_feat = feat_mat[tuple(list(train_ij.T))]

    n_spy = int(len(pos_train_ij) * 0.1)  # 0.1 is the ratio of spy

    spy_ij_idx = np.random.choice(len(pos_train_ij), n_spy, replace=False)
    spy_ij = pos_train_ij[spy_ij_idx]

    adj_np_hidden = adj_np.copy()
    adj_np_hidden[tuple(list(spy_ij.T))] = 0
    train_label = adj_np_hidden[tuple(list(train_ij.T))]
    
    regressor = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    regressor.fit(train_feat, train_label)
    
    with open(rf".\classifier\rfclf_f85_e500_{i}.pickle", "wb") as f:
        pickle.dump(regressor, f)

    # with open(rf".\classifier\rfclf_f85_e500_{i}.pickle", "rb") as f:
    #     regressor = pickle.load(f)

    prob_mat = np.zeros_like(adj_np)
    train_prob = regressor.predict_proba(train_feat)[:, 1]
    prob_mat[tuple(list(train_ij.T))] = train_prob
    spy_prob = prob_mat[tuple(list(spy_ij.T))]
    prob_spymin = prob_mat[tuple(list(spy_ij.T))].min()
    pos_prob = prob_mat[tuple(list(pos_train_ij.T))]

    pos_prob_thresh = np.sort(spy_prob)[int(len(spy_prob) * 0.05)]
    rn_ij = np.argwhere(prob_mat < pos_prob_thresh)
    rn_ij_list.append(rn_ij)

with open("rn_ij_list.pickle", "wb") as f:
    pickle.dump(rn_ij_list, f)
