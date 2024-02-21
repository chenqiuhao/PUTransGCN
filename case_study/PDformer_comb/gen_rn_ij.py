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


with open(r"../data/p_gip.pickle", "rb") as f:
    p_gip = pickle.load(f)
with open(r"../data/d_gip.pickle", "rb") as f:
    d_gip = pickle.load(f)

pos_ij = np.argwhere(adj_np == 1)
unlabelled_ij = np.argwhere(adj_np == 0)

train_ij = np.vstack((pos_ij, unlabelled_ij))
train_feat = feat_mat[tuple(list(train_ij.T))]

n_spy = int(len(pos_ij) * 0.1)  # 0.1 is the ratio of spy

spy_ij_idx = np.random.choice(len(pos_ij), n_spy, replace=False)
spy_ij = pos_ij[spy_ij_idx]

adj_np_hidden = adj_np.copy()
adj_np_hidden[tuple(list(spy_ij.T))] = 0
train_label = adj_np_hidden[tuple(list(train_ij.T))]

regressor = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
regressor.fit(train_feat, train_label)

prob_mat = np.zeros_like(adj_np)
train_prob = regressor.predict_proba(train_feat)[:, 1]
prob_mat[tuple(list(train_ij.T))] = train_prob
spy_prob = prob_mat[tuple(list(spy_ij.T))]
prob_spymin = prob_mat[tuple(list(spy_ij.T))].min()
pos_prob = prob_mat[tuple(list(pos_ij.T))]

pos_prob_thresh = np.sort(spy_prob)[int(len(spy_prob) * 0.05)]
rn_ij_spy = np.argwhere(prob_mat < pos_prob_thresh)

# pu bagging
T = 10
cnt = np.zeros_like(adj_np)
for t in range(T):
    print(t)
    unlabelled_train_ij_t_idx = np.random.choice(
        np.arange(len(unlabelled_ij)), replace=False, size=len(pos_ij)
    )
    unlabelled_train_ij_t = unlabelled_ij[unlabelled_train_ij_t_idx]
    rest_unlabelled_train_ij_t_idx = np.setdiff1d(
        np.arange(len(unlabelled_ij)), unlabelled_train_ij_t_idx
    )
    rest_unlabelled_train_ij_t = unlabelled_ij[rest_unlabelled_train_ij_t_idx]

    train_ij = np.vstack((pos_ij, unlabelled_train_ij_t))
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

non_nan_indexes = np.nonzero(~np.isnan(fnl_score))
non_nan_values = fnl_score[non_nan_indexes]
non_nan_ij = np.transpose(non_nan_indexes)

pos_prob = regressor.predict_proba(feat_mat[tuple(list(pos_ij.T))])[:, 1]
pos_prob_min = pos_prob.min()

sorted_nums = sorted(enumerate(non_nan_values), key=lambda x: x[1])
neg_nums = list(filter(lambda x: x[1] < pos_prob_min, sorted_nums))
idx = [i[0] for i in neg_nums]

rn_idx = idx[int(len(idx) / 3): int(len(idx) / 3 * 2)]
rn_ij_pu = non_nan_ij[rn_idx]

# two step

train_ij = np.vstack((pos_ij, unlabelled_ij))
train_feat = feat_mat[tuple(list(train_ij.T))]
train_label = adj_np[tuple(list(train_ij.T))]
ys = (2 * train_label - 1).copy()
regressor1 = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
regressor1.fit(train_feat, train_label)

# with open(rf".\classifier\rfclf_f85_e500_{i}.pickle", "wb") as f:
#     pickle.dump(regressor1, f)

train_prob = regressor1.predict_proba(train_feat)[:, 1]

range_pos = [min(train_prob * (ys > 0)), max(train_prob * (ys > 0))]

p_new = ys[(ys < 0) & (train_prob >= range_pos[1])]
n_new = ys[(ys < 0) & (train_prob <= range_pos[0])]
ys[(ys < 0) & (train_prob >= range_pos[1])] = 1
ys[(ys < 0) & (train_prob <= range_pos[0])] = 0

regressor2 = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
for j in range(5):
    print("Iteration: ", j, "len(p_new)", len(p_new), "len(n_new)", len(n_new))
    if len(p_new) + len(n_new) == 0:# and i > 0:
        break

    regressor2.fit(train_feat, ys)

    train_prob = regressor2.predict_proba(train_feat)[:, -1]

    # Find the range of scores given to positive data points
    range_pos = [min(train_prob * (ys > 0)), max(train_prob * (ys > 0))]

    # Repeat step 1
    p_new = ys[(ys < 0) & (train_prob >= range_pos[1])]
    n_new = ys[(ys < 0) & (train_prob <= range_pos[0])]
    ys[(ys < 0) & (train_prob >= range_pos[1])] = 1
    ys[(ys < 0) & (train_prob <= range_pos[0])] = 0

rn_ij_two = train_ij[ys == 0]
rn_ij = np.concatenate((rn_ij_spy, rn_ij_pu, rn_ij_two))

with open("rn_ij.pickle", "wb") as f:
    pickle.dump(rn_ij, f)
