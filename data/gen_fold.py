import pandas as pd
import numpy as np
import math
import pickle


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


def Getgauss_miRNA(adjacentmatrix, nm):
    """
    MiRNA Gaussian interaction profile kernels similarity
    """
    KM = np.zeros((nm, nm))

    gamaa = 1
    sumnormm = 0
    for i in range(nm):
        normm = np.linalg.norm(adjacentmatrix[i]) ** 2
        sumnormm = sumnormm + normm
    gamam = gamaa / (sumnormm / nm)

    for i in range(nm):
        for j in range(nm):
            KM[i, j] = math.exp(
                -gamam * (np.linalg.norm(adjacentmatrix[i] - adjacentmatrix[j]) ** 2)
            )
    return KM


def Getgauss_disease(adjacentmatrix, nd):
    """
    Disease Gaussian interaction profile kernels similarity
    """
    KD = np.zeros((nd, nd))
    gamaa = 1
    sumnormd = 0
    for i in range(nd):
        normd = np.linalg.norm(adjacentmatrix[:, i]) ** 2
        sumnormd = sumnormd + normd
    gamad = gamaa / (sumnormd / nd)

    for i in range(nd):
        for j in range(nd):
            KD[i, j] = math.exp(
                -(
                    gamad
                    * (np.linalg.norm(adjacentmatrix[:, i] - adjacentmatrix[:, j]) ** 2)
                )
            )
    return KD


seed_everything(42)
adj_df = pd.read_csv(r"adj.csv", index_col=0)
adj = adj_df.values
num_p, num_d = adj.shape

pos_ij = np.argwhere(adj == 1)
unlabelled_ij = np.argwhere(adj == 0)
np.random.shuffle(pos_ij)
np.random.shuffle(unlabelled_ij)
k_fold = 5
pos_ij_5fold = np.array_split(pos_ij, k_fold)
unlabelled_ij_5fold = np.array_split(unlabelled_ij, k_fold)

fold_cnt = 0

pos_train_ij_list = []
pos_test_ij_list = []
unlabelled_train_ij_list = []
unlabelled_test_ij_list = []
p_gip_list = []
d_gip_list = []

for i in range(k_fold):
    extract_idx = list(range(k_fold))
    extract_idx.remove(i)

    pos_train_ij = np.vstack([pos_ij_5fold[idx] for idx in extract_idx])
    pos_test_ij = pos_ij_5fold[i]

    unlabelled_train_ij = np.vstack([unlabelled_ij_5fold[idx] for idx in extract_idx])
    unlabelled_test_ij = unlabelled_ij_5fold[i]

    A = np.zeros_like(adj)
    A[tuple(list(pos_train_ij.T))] = 1

    p_gip = Getgauss_miRNA(A, num_p)
    d_gip = Getgauss_disease(A, num_d)
    # p_gip = np.ones((num_p, num_p))
    # d_gip = np.ones((num_d, num_d))

    pos_train_ij_list.append(pos_train_ij)
    pos_test_ij_list.append(pos_test_ij)
    unlabelled_train_ij_list.append(unlabelled_train_ij)
    unlabelled_test_ij_list.append(unlabelled_test_ij)
    p_gip_list.append(p_gip)
    d_gip_list.append(d_gip)

    fold_cnt = fold_cnt + 1

fold_info = {
    "pos_train_ij_list": pos_train_ij_list,
    "pos_test_ij_list": pos_test_ij_list,
    "unlabelled_train_ij_list": unlabelled_train_ij_list,
    "unlabelled_test_ij_list": unlabelled_test_ij_list,
    "p_gip_list": p_gip_list,
    "d_gip_list": d_gip_list,
}
with open("fold_info.pickle", "wb") as f:
    pickle.dump(fold_info, f)

with open(r"fold_info.pickle", "rb") as f:
    fold_info = pickle.load(f)
