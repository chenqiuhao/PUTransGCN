import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
import os
import copy

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

adj_df = pd.read_csv(r"adj.csv", index_col=0)
adj_np = adj_df.values
num_p, num_d = adj_np.shape

p_sim_smith = pd.read_csv(r"p2p_smith.csv", index_col=0).values
d_sim_do = pd.read_csv(r"d2d_do.csv", index_col=0).values

feat_mat = np.zeros((num_p, num_d, (num_p + num_d)))

for i in range(num_p):
    for j in range(num_d):
        feat_mat[i, j] = np.append(p_sim_smith[i], d_sim_do[j])
# feat_mat = feat_mat[:,:,0:3]

with open(r"fold_info.pickle", "rb") as f:
    fold_info = pickle.load(f)

pos_train_ij_list = fold_info["pos_train_ij_list"]
pos_test_ij_list = fold_info["pos_test_ij_list"]
unlabelled_train_ij_list = fold_info["unlabelled_train_ij_list"]
unlabelled_test_ij_list = fold_info["unlabelled_test_ij_list"]
p_gip_list = fold_info["p_gip_list"]
d_gip_list = fold_info["d_gip_list"]

path = "weight_adj/"
if not os.path.exists(path):
    os.makedirs(path)

for i in range(5):
    pos_train_ij = pos_train_ij_list[i]
    pos_test_ij = pos_test_ij_list[i]
    unlabelled_train_ij = unlabelled_train_ij_list[i]
    unlabelled_test_ij = unlabelled_test_ij_list[i]

    test_ij = np.vstack((pos_test_ij, unlabelled_train_ij, unlabelled_test_ij))
    test_feat = feat_mat[tuple(list(test_ij.T))]

    score = np.zeros_like(adj_np)
    for j in range(5):
        print(f'j{j}')
        unlabelled_train_ij_t_idx = np.random.choice(
            np.arange(len(unlabelled_train_ij)), replace=False, size=len(pos_train_ij)
        )
        unlabelled_train_ij_t = unlabelled_train_ij[unlabelled_train_ij_t_idx]

        train_ij = np.vstack((pos_train_ij, unlabelled_train_ij_t))
        train_feat = feat_mat[tuple(list(train_ij.T))]
        train_label = adj_np[tuple(list(train_ij.T))]


        rf_model = RandomForestClassifier(
            n_estimators=100, max_leaf_nodes=10, n_jobs=-1, max_features=0.2
        )
        rf_model.fit(train_feat, train_label)
        score_rf = rf_model.predict_proba(test_feat)[:, 1]
        score[tuple(list(test_ij.T))] += score_rf
        print('0')

        gbdt_model = GradientBoostingClassifier()
        gbdt_model.fit(train_feat, train_label)
        score_gbdt = gbdt_model.predict_proba(test_feat)[:, 1]
        score[tuple(list(test_ij.T))] += score_gbdt
        print('1')

        # svm_model = svm.SVC(kernel="rbf", gamma=20, probability=True)
        # svm_model.fit(train_feat, train_label)
        # score_gbdt = svm_model.predict_proba(test_feat)[:, 1]
        # score[tuple(list(test_ij.T))] += score_gbdt
        # print('2')

    score = score / 10
    score[tuple(list(pos_train_ij.T))] = 1

    # weight_adj = adj_df.copy()
    # weight_adj.value = score
    weight_adj = pd.DataFrame(score, index=adj_df.index, columns=adj_df.columns)

    weight_adj.to_csv(path + f"weight_adj_{i}.csv")
