import numpy as np
from utils import *
import torch
from model import *
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import warnings
import pickle
import os

warnings.filterwarnings("ignore")
seed_everything(42)
device = torch.device("cuda")

def cal_jaccard(adj):
    num_d = adj.shape[1]
    d_sim_jaccard = np.zeros((num_d, num_d))

    def jaccard_binary(x, y):
        intersection = np.logical_and(x, y)
        union = np.logical_or(x, y)
        similarity = intersection.sum() / float(union.sum())
        return similarity

    for i in range(num_d):
        for j in range(num_d):
            d_sim_jaccard[i, j] = jaccard_binary(adj[:, i], adj[:, j])
    return d_sim_jaccard


adj_df = pd.read_csv(r"..\data\adj.csv", index_col=0)
adj_np = adj_df.values
num_p, num_d = adj_np.shape

# Since the Levenshtein distance cannot calculate values between 0 and 1,
# the paper did not provide a regularization method,
# so Smith-Waterman similarity was used instead.
p_sim_smith = pd.read_csv(r"..\data\p2p_smith.csv", index_col=0).values

with open(r"../data/p_gip.pickle", "rb") as f:
    p_gip = pickle.load(f)
with open(r"../data/d_gip.pickle", "rb") as f:
    d_gip = pickle.load(f)

pos_ij = np.argwhere(adj_np == 1)
unlabelled_ij = np.argwhere(adj_np == 0)
unlabelled_ij = unlabelled_ij[: len(pos_ij)]

d_sim_do = pd.read_csv(r"..\data\d2d_do.csv", index_col=0).values

adj = torch.FloatTensor(adj_np).to(device)

lr = 0.01
weight_decay = 0.0
num_epochs = 200

class MaskedBCELoss(nn.BCELoss):
    def forward(self, pred, adj, train_mask):
        self.reduction = "none"
        unweighted_loss = super(MaskedBCELoss, self).forward(pred, adj)
        train_loss = (unweighted_loss * train_mask).sum()
        return train_loss


def fit(model, p_M, d_M, train_mask, lr, num_epochs):
    # optimizer = torch.optim.RMSprop(model.parameters(), lr, weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = MaskedBCELoss()

    # test_idx = torch.argwhere(torch.ones_like(test_mask) == 1)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(p_M, d_M)
        train_loss = loss(pred, adj, train_mask)
        train_loss.backward()
        optimizer.step()
    model.eval()
    pred = model(p_M, d_M)
    return pred


A_pos = np.zeros_like(adj_np)
A_pos[tuple(list(pos_ij.T))] = 1
d_jaccard = cal_jaccard(A_pos)
d_jaccard = np.nan_to_num(d_jaccard, nan=0)

p_M = (p_sim_smith + p_gip) / 2
d_M = (d_sim_do + d_gip + d_jaccard) / 3

A_corner_np = np.zeros_like(adj_np)
A_corner_np[tuple(list(pos_ij.T))] = 1

# train_mask_np = np.ones_like(adj_np)
train_mask_np = np.zeros_like(adj_np)
train_mask_np[tuple(list(pos_ij.T))] = 1
train_mask_np[tuple(list(unlabelled_ij.T))] = 1

A_corner = torch.FloatTensor(A_corner_np).to(device)
p_M = torch.FloatTensor(p_M).to(device)
d_M = torch.FloatTensor(d_M).to(device)
train_mask = torch.FloatTensor(train_mask_np).to(device)

model = GBNN(num_p, num_d).to(device)
pred = fit(model, p_M, d_M, train_mask, lr, num_epochs)

scores = pred[tuple(list(unlabelled_ij.T))].cpu().detach().numpy()

pred_np = pred.cpu().detach().numpy()
np.savetxt('pred_np.csv', pred_np, delimiter=",")

max_allocated_memory = torch.cuda.max_memory_allocated()
print(f"最大已分配内存量: {max_allocated_memory / 1024 ** 2} MB")

