import time
from sklearn import tree, svm
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import obonet
import networkx as nx
import math
from utils import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")
seed_everything(42)

d_sim_do = pd.read_csv(r"..\data\d2d_do.csv", index_col=0).values
adj_df = pd.read_csv(r"..\data\adj.csv", index_col=0)
adj = adj_df.values
p_sim_smith = pd.read_csv(r"..\data\p2p_smith.csv", index_col=0).values
num_p, num_d = adj.shape

logger = Logger(5)
fold_cnt = 0
T = 30
pca = PCA(n_components=200)

with open(r"../data/p_gip.pickle", "rb") as f:
    p_gip = pickle.load(f)
with open(r"../data/d_gip.pickle", "rb") as f:
    d_gip = pickle.load(f)

pos_ij = np.argwhere(adj == 1)
unlabelled_ij = np.argwhere(adj == 0)

p_sim = (p_sim_smith + p_gip) / 2
d_sim = (d_sim_do + d_gip) / 2

feat_mat = np.zeros((num_p, num_d, (num_p + num_d)), dtype="float32")

for i in range(num_p):
    for j in range(num_d):
        feat_mat[i, j] = np.append(p_sim[i], d_sim[j])

pred = np.zeros_like(adj)
cnt = np.zeros_like(adj)

for i in range(T):
    a = time.time()
    unlabelled_train_ij_t_idx = np.random.choice(
        np.arange(len(unlabelled_ij)), replace=False, size=len(pos_ij)
    )
    unlabelled_train_ij_t = unlabelled_ij[unlabelled_train_ij_t_idx]

    train_ij = np.vstack((pos_ij, unlabelled_train_ij_t))
    train_feat = feat_mat[tuple(list(train_ij.T))]
    train_label = adj[tuple(list(train_ij.T))]
    pca_train_feat = pca.fit_transform(train_feat)

    test_ij = np.vstack((unlabelled_ij))
    test_feat = feat_mat[tuple(list(test_ij.T))]
    pca_test_feat = pca.transform(test_feat)

    # regressor = RandomForestClassifier(
    #     n_estimators=80, max_features=0.2, random_state=42
    # )

    regressor = svm.SVC(probability=True)

    # regressor = tree.DecisionTreeClassifier(
    #     criterion="entropy", max_features=0.2, random_state=42
    # )
    regressor.fit(pca_train_feat, train_label)

    pred[tuple(list(test_ij.T))] = (
        pred[tuple(list(test_ij.T))] + regressor.predict_proba(pca_test_feat)[:, 1]
    )
    cnt[tuple(list(test_ij.T))] = cnt[tuple(list(test_ij.T))] + 1

    b = time.time()
    print(i, b - a)
fnl_score = pred / cnt
fnl_score[np.isnan(fnl_score)]=1
np.savetxt('pred_np_svm.csv', fnl_score, delimiter=",")

# fnl_score_df = pd.DataFrame(fnl_score, index=adj_df.index, columns=adj_df.columns)
# fnl_score_df.to_csv(f"pred_svm_{fold_cnt}.csv")
# score = fnl_score[~np.isnan(fnl_score)]
# true = adj[~np.isnan(fnl_score)]

