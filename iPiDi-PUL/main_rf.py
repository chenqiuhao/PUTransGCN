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


with open(r"../data/fold_info.pickle", "rb") as f:
    fold_info = pickle.load(f)

pos_train_ij_list = fold_info["pos_train_ij_list"]
pos_test_ij_list = fold_info["pos_test_ij_list"]
unlabelled_train_ij_list = fold_info["unlabelled_train_ij_list"]
unlabelled_test_ij_list = fold_info["unlabelled_test_ij_list"]
p_gip_list = fold_info["p_gip_list"]
d_gip_list = fold_info["d_gip_list"]


for i in range(5):
    pos_train_ij = pos_train_ij_list[i]
    pos_test_ij = pos_test_ij_list[i]
    unlabelled_train_ij = unlabelled_train_ij_list[i]
    unlabelled_test_ij = unlabelled_test_ij_list[i]
    p_gip = p_gip_list[i]
    d_gip = d_gip_list[i]

    p_sim = (p_sim_smith + p_gip) / 2
    d_sim = (d_sim_do + d_gip) / 2

    feat_mat = np.zeros((num_p, num_d, (num_p + num_d)))

    for i in range(num_p):
        for j in range(num_d):
            feat_mat[i, j] = np.append(p_sim[i], d_sim[j])

    pred = np.zeros_like(adj)
    cnt = np.zeros_like(adj)

    for i in range(T):
        a = time.time()
        unlabelled_train_ij_t_idx = np.random.choice(
            np.arange(len(unlabelled_train_ij)), replace=False, size=len(pos_train_ij)
        )
        unlabelled_train_ij_t = unlabelled_train_ij[unlabelled_train_ij_t_idx]

        train_ij = np.vstack((pos_train_ij, unlabelled_train_ij_t))
        train_feat = feat_mat[tuple(list(train_ij.T))]
        train_label = adj[tuple(list(train_ij.T))]
        pca_train_feat = pca.fit_transform(train_feat)

        test_ij = np.vstack((pos_test_ij, unlabelled_test_ij))
        test_feat = feat_mat[tuple(list(test_ij.T))]
        test_label = adj[tuple(list(test_ij.T))]
        pca_test_feat = pca.transform(test_feat)

        regressor = RandomForestClassifier(
            n_estimators=80, max_features=0.2, n_jobs=-1, random_state=42
        )

        # regressor = svm.SVR()

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
    fnl_score_df = pd.DataFrame(fnl_score, index=adj_df.index, columns=adj_df.columns)
    fnl_score_df.to_csv(f"pred_rf_{fold_cnt}.csv")
    score = fnl_score[~np.isnan(fnl_score)]
    true = adj[~np.isnan(fnl_score)]
    logger.update(0, fold_cnt, true, score, 0, 0)
    fold_cnt += 1

logger.save("rf")
