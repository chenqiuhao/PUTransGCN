import os
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt

# plot_dir = [
#     "./PDformer_spy",
#     "./PDformer_two_step",
#     "./PDformer_pu_bagging",
#     "./PDformer_all",
# ]
plot_dir = [
    "./PDformer_spy",
    "./ETGPDA",
    "./iPiDi-PUL/iPiDi-PUL(DT)",
    "./iPiDi-PUL/iPiDi-PUL(SVM)",
    "./iPiDi-PUL/iPiDi-PUL(RF)",
    "./iPiDA-GCN",
    "./iPiDA-SWGCN",
    "./piRDA",
]
adj_np = pd.read_csv(r".\data\adj.csv", index_col=0).values

with open(r"./data/fold_info.pickle", "rb") as f:
    fold_info = pickle.load(f)

pos_train_ij_list = fold_info["pos_train_ij_list"]
pos_test_ij_list = fold_info["pos_test_ij_list"]
unlabelled_train_ij_list = fold_info["unlabelled_train_ij_list"]
unlabelled_test_ij_list = fold_info["unlabelled_test_ij_list"]
p_gip_list = fold_info["p_gip_list"]
d_gip_list = fold_info["d_gip_list"]

plt.figure(1)
plt.grid()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.tight_layout()

plt.figure(2)
plt.grid()
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()

for dir in plot_dir:
    scores_dir = dir + "/scores"
    model_name = os.path.basename(dir)
    fold = 1

    pos_test_ij = pos_test_ij_list[fold]
    unlabelled_test_ij = unlabelled_test_ij_list[fold]

    # test_ij = np.vstack((pos_test_ij, unlabelled_test_ij))
    test_mask_np = np.zeros_like(adj_np)
    test_mask_np[tuple(list(pos_test_ij.T))] = 1
    test_mask_np[tuple(list(unlabelled_test_ij.T))] = 1
    if "piRDA" in dir:
        test_ij = np.concatenate((pos_test_ij, unlabelled_test_ij), axis=0)
    else:
        test_ij = np.argwhere(test_mask_np == 1)

    labels = adj_np[tuple(list(test_ij.T))]

    score_path = scores_dir + f"/f{fold}_e199_scores.npy"
    score = np.load(score_path)

    fpr, tpr, thresholds_ = metrics.roc_curve(labels, score)
    auc = metrics.auc(fpr, tpr)
    plt.figure(1)
    # plt.title("AUC")
    plt.plot(fpr, tpr, label=f"{model_name} AUC = %0.3f" % auc)
    plt.legend()

    precisions, recalls, thresholds = metrics.precision_recall_curve(labels, score)
    aupr = metrics.auc(recalls, precisions)
    plt.figure(2)
    # plt.title("AUPR")
    plt.plot(recalls, precisions, label=f"{model_name} AUPR = %0.3f" % aupr)
    plt.legend()

plt.figure(1)
plt.savefig("ROC_curve.png", dpi=300)
plt.figure(2)
plt.savefig("PR_curve.png", dpi=300)
plt.show()
