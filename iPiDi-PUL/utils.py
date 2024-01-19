import random
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import torch
import itertools
from sklearn import metrics


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


def rwr(A, restart_prob):
    """
    Random Walk with Restart (RWR) on similarity network.
    :param A: n x n, similarity matrix
    :param restart_prob: probability of restart
    :return: n x n, steady-state probability
    """
    n = A.shape[0]
    A = (A + A.T) / 2
    A = A - np.diag(np.diag(A))
    A = A + np.diag(sum(A) == 0)
    P = A / sum(A)
    Q = np.linalg.inv(np.eye(n) - (1 - restart_prob) * P) @ (restart_prob * np.eye(n))
    return Q


def get_pos_neg_ij(adj):
    num_p, num_d = adj.shape
    positive_ij = []
    negative_ij = []
    for i in range(num_p):
        for j in range(num_d):
            label = adj[i, j]
            if label == 1:
                positive_ij.append((i, j))
            elif label == 0:
                negative_ij.append((i, j))
    pos_ij = np.array(positive_ij)
    neg_ij = np.array(negative_ij)
    return pos_ij, neg_ij


def gen_folds(adj, T):
    pos_ij = np.argwhere(adj == 1)
    neg_ij = np.argwhere(adj == 0)

    positive_idx = np.array(range(0, len(pos_ij)))
    np.random.shuffle(positive_idx)
    negative_idx = np.array(range(0, len(neg_ij)))
    np.random.shuffle(negative_idx)

    kf = KFold(n_splits=5)
    for train, test in kf.split(positive_idx):
        neg_train_ij_t = []
        neg_test_ij_t = []
        pos_train_ij = pos_ij[positive_idx[train]]
        pos_test_ij = pos_ij[positive_idx[test]]
        for i in range(T):
            rand_neg_idx = random.sample(range(len(neg_ij)), len(train))
            complementaryset = list(set(range(len(neg_ij))) - (set(rand_neg_idx)))
            neg_train_ij = neg_ij[negative_idx[rand_neg_idx]]
            neg_test_ij = neg_ij[negative_idx[complementaryset]]
            neg_train_ij_t.append(neg_train_ij)
            neg_test_ij_t.append(neg_test_ij)

        positive_train_mask = np.zeros_like(adj, dtype=int)
        positive_train_mask[tuple(list(pos_train_ij.T))] = 1

        yield positive_train_mask, pos_train_ij, pos_test_ij, neg_train_ij_t, neg_test_ij_t


def matrix(a, b, match_score=3, gap_cost=2):
    H = np.zeros((len(a) + 1, len(b) + 1), int)

    for i, j in itertools.product(range(1, H.shape[0]), range(1, H.shape[1])):
        match = H[i - 1, j - 1] + (
            match_score if a[i - 1] == b[j - 1] else -match_score
        )
        delete = H[i - 1, j] - gap_cost
        insert = H[i, j - 1] - gap_cost
        H[i, j] = max(match, delete, insert, 0)
    return H


def traceback(H, b, b_="", old_i=0):
    # flip H to get index of **last** occurrence of H.max() with np.argmax()
    H_flip = np.flip(np.flip(H, 0), 1)
    i_, j_ = np.unravel_index(H_flip.argmax(), H_flip.shape)
    i, j = np.subtract(
        H.shape, (i_ + 1, j_ + 1)
    )  # (i, j) are **last** indexes of H.max()
    if H[i, j] == 0:
        return b_, j
    b_ = b[j - 1] + "-" + b_ if old_i - i > 1 else b[j - 1] + b_
    return traceback(H[0:i, 0:j], b, b_, i)


def smith_waterman(a, b, match_score=3, gap_cost=2):
    a, b = a.upper(), b.upper()
    H = matrix(a, b, match_score, gap_cost)
    b_, pos = traceback(H, b)
    return pos, pos + len(b_)


# class Logger:
#     def __init__(self, total_fold):
#         def gen_dict():
#             return {
#                 "epoch": [],
#                 "f1_score": [],
#                 "auc": [],
#                 "aupr": [],
#                 "threshold": [],
#                 "recall": [],
#                 "precision": [],
#                 "acc": [],
#                 "specificity": [],
#                 "mcc": [],
#                 "train_loss": [],
#                 "test_loss": [],
#             }
#
#         self.df = [gen_dict() for i in range(total_fold)]
#
#     def evaluate(self, true, pred):
#         labels = true
#         scores = pred
#
#         fpr, tpr, thresholds_ = metrics.roc_curve(labels, scores)
#         auc = metrics.auc(fpr, tpr)
#
#         precisions, recalls, thresholds = metrics.precision_recall_curve(labels, scores)
#         aupr = metrics.auc(recalls, precisions)
#         # aupr2 = metrics.average_precision_score(labels, scores)
#         num = 2 * recalls * precisions
#         den = recalls + precisions
#         den[den == 0] = 100
#         f1_scores = num / den
#         f1_score = f1_scores.max()
#         f1_score_idx = np.argmax(f1_scores)
#         threshold = thresholds[np.argmax(f1_scores)]
#         precision = precisions[f1_score_idx]
#         recall = recalls[f1_score_idx]
#         bi_scores = scores.copy()
#         bi_scores[bi_scores < threshold] = 0
#         bi_scores[bi_scores >= threshold] = 1
#         acc = metrics.accuracy_score(labels, bi_scores)
#         tn, fp, fn, tp = metrics.confusion_matrix(labels, bi_scores).ravel()
#         specificity = tn / (tn + fp)
#         # mcc = metrics.matthews_corrcoef(labels, bi_scores)
#         mcc = (tp * tn - fp * fn) / np.sqrt(
#             (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
#         )
#         return tuple(
#             np.round(
#                 [
#                     f1_score,
#                     auc,
#                     aupr,
#                     threshold,
#                     recall,
#                     precision,
#                     acc,
#                     specificity,
#                     mcc,
#                 ],
#                 3,
#             )
#         )
#
#     def update(self, fold, epoch, test_label, pred, train_loss, test_loss):
#         (
#             f1_score,
#             auc,
#             aupr,
#             threshold,
#             recall,
#             precision,
#             acc,
#             specificity,
#             mcc,
#         ) = self.evaluate(test_label, pred)
#         self.df[fold]["epoch"].append(epoch)
#         self.df[fold]["f1_score"].append(f1_score)
#         self.df[fold]["auc"].append(auc)
#         self.df[fold]["aupr"].append(aupr)
#         self.df[fold]["threshold"].append(threshold)
#         self.df[fold]["recall"].append(recall)
#         self.df[fold]["precision"].append(precision)
#         self.df[fold]["acc"].append(acc)
#         self.df[fold]["specificity"].append(specificity)
#         self.df[fold]["mcc"].append(mcc)
#         self.df[fold]["train_loss"].append(train_loss)
#         self.df[fold]["test_loss"].append(test_loss)
#         print(
#             f"fold:{fold}, epoch:{epoch}, f1_score: {f1_score}, auc: {auc}, aupr: {aupr}, acc: {acc}, "
#             f"specificity: {specificity}, threshold: {threshold}, recall: {recall}, "
#             f"precision: {precision}, mcc: {mcc}, train_loss: {int(train_loss)}, test_loss: {int(test_loss)}"
#         )
#
#     def save(self, name):
#         with pd.ExcelWriter(f"result_{name}.xlsx") as writer:
#             for fold in range(len(self.df)):
#                 pd.DataFrame(self.df[fold]).to_excel(
#                     writer, sheet_name=f"fold{fold}", index=False
#                 )
class Logger:
    def __init__(self, total_fold):
        def gen_dict():
            return {
                "epoch": [],
                "f1_score": [],
                "f2_score": [],
                "rank_idx": [],
                "auc": [],
                "aupr": [],
                "threshold": [],
                "recall": [],
                "precision": [],
                "acc": [],
                "specificity": [],
                "mcc": [],
                "train_loss": [],
                "test_loss": [],
            }

        self.df = [gen_dict() for i in range(total_fold)]

    def evaluate(self, true, pred):
        labels = true
        scores = pred

        combined = list(zip(labels, scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        labels_sorted, scores_sorted = zip(*combined)

        indices = np.arange(1, len(labels) + 1)[np.array(labels_sorted) == 1]
        n_test = len(labels)
        n_test_p = sum(labels == 1)
        rank_idx = indices.sum() / n_test / n_test_p

        fpr, tpr, thresholds_ = metrics.roc_curve(labels, scores)
        auc = metrics.auc(fpr, tpr)

        precisions, recalls, thresholds = metrics.precision_recall_curve(labels, scores)
        aupr = metrics.auc(recalls, precisions)
        # aupr2 = metrics.average_precision_score(labels, scores)
        num1 = 2 * recalls * precisions
        den1 = recalls + precisions
        den1[den1 == 0] = 100
        f1_scores = num1 / den1
        f1_score = f1_scores.max()
        beta2 = 2
        num2 = (1 + beta2**2) * recalls * precisions
        den2 = recalls + precisions * beta2**2
        den2[den2 == 0] = 100
        f2_scores = num2 / den2
        f2_score = f2_scores.max()
        f2_score_idx = np.argmax(f2_scores)
        threshold = thresholds[np.argmax(f2_scores)]
        precision = precisions[f2_score_idx]
        recall = recalls[f2_score_idx]
        bi_scores = scores.copy()
        bi_scores[bi_scores < threshold] = 0
        bi_scores[bi_scores >= threshold] = 1
        acc = metrics.accuracy_score(labels, bi_scores)
        tn, fp, fn, tp = metrics.confusion_matrix(labels, bi_scores).ravel()
        specificity = tn / (tn + fp)
        # mcc = metrics.matthews_corrcoef(labels, bi_scores)
        mcc = (tp * tn - fp * fn) / np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        return tuple(
            np.round(
                [
                    f1_score,
                    f2_score,
                    rank_idx,
                    auc,
                    aupr,
                    threshold,
                    recall,
                    precision,
                    acc,
                    specificity,
                    mcc,
                ],
                6,
            )
        )

    def update(self, fold, epoch, test_label, pred, train_loss, test_loss):
        (
            f1_score,
            f2_score,
            rank_idx,
            auc,
            aupr,
            threshold,
            recall,
            precision,
            acc,
            specificity,
            mcc,
        ) = self.evaluate(test_label, pred)
        self.df[fold]["epoch"].append(epoch)
        self.df[fold]["f1_score"].append(f1_score)
        self.df[fold]["f2_score"].append(f2_score)
        self.df[fold]["rank_idx"].append(rank_idx)
        self.df[fold]["auc"].append(auc)
        self.df[fold]["aupr"].append(aupr)
        self.df[fold]["threshold"].append(threshold)
        self.df[fold]["recall"].append(recall)
        self.df[fold]["precision"].append(precision)
        self.df[fold]["acc"].append(acc)
        self.df[fold]["specificity"].append(specificity)
        self.df[fold]["mcc"].append(mcc)
        self.df[fold]["train_loss"].append(train_loss)
        self.df[fold]["test_loss"].append(test_loss)
        print(
            f"fold:{fold}, epoch:{epoch}, f1: {f1_score}, f2: {f2_score}, rank_idx: {rank_idx}, auc: {auc}, "
            f"aupr: {aupr}, acc: {acc}, specificity: {specificity}, threshold: {threshold}, recall: {recall}, "
            f"precision: {precision}, mcc: {mcc}, train_loss: {int(train_loss)}, test_loss: {int(test_loss)}"
        )

    def save(self, name):
        with pd.ExcelWriter(f"{name}.xlsx") as writer:
            for fold in range(len(self.df)):
                pd.DataFrame(self.df[fold]).to_excel(
                    writer, sheet_name=f"fold{fold}", index=False
                )
