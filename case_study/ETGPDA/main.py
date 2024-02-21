import numpy as np
from utils import *
import torch
from model import *
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import pickle

seed_everything(42)
device = torch.device("cuda")

adj_df = pd.read_csv(r"..\data\adj.csv", index_col=0)
adj_np = adj_df.values
num_p, num_d = adj_np.shape
adj = torch.FloatTensor(adj_np).to(device)

p_sim_needleman = pd.read_csv(r"..\data\p2p_needleman.csv", index_col=0).values
d_sim_do = pd.read_csv(r"..\data\d2d_do.csv", index_col=0).values

adj_dp = 0.6
f_dp = 0.4

lr = 0.01
weight_decay = 0.8
num_epochs = 200

class MaskedBCELoss(nn.BCELoss):
    def forward(self, pred, adj, train_mask):
        self.reduction = "none"
        unweighted_loss = super(MaskedBCELoss, self).forward(pred, adj)
        train_loss = (unweighted_loss * train_mask).sum()
        return train_loss


def fit(model, adj, train_mask, lr, num_epochs):
    loss = MaskedBCELoss()
    # optimizer = torch.optim.RMSprop(model.parameters(), lr, weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # test_idx = torch.argwhere(torch.ones_like(test_mask) == 1)

    for epoch in range(num_epochs):
        model.train()
        model.zero_grad()
        pred = model()
        train_loss = loss(pred, adj, train_mask)
        print(train_loss)
        train_loss.backward()
        optimizer.step()

    model.eval()
    pred = model()
    return pred

with open(r"../data/p_gip.pickle", "rb") as f:
    p_gip = pickle.load(f)
with open(r"../data/d_gip.pickle", "rb") as f:
    d_gip = pickle.load(f)

pos_ij = np.argwhere(adj_np == 1)
unlabelled_ij = np.argwhere(adj_np == 0)

p_sim_needleman[p_sim_needleman == 0] = p_gip[p_sim_needleman == 0]
d_sim_do[d_sim_do == 0] = d_gip[d_sim_do == 0]
p_sim_np = p_sim_needleman
d_sim_np = d_sim_do

A_corner_np = np.zeros_like(adj_np)
A_corner_np[tuple(list(pos_ij.T))] = 1
A_np = np.concatenate(
    (
        np.concatenate((np.eye(num_p), A_corner_np), axis=1),
        np.concatenate(((A_corner_np).T, np.eye(num_d)), axis=1),
    ),
    axis=0,
)

train_mask_np = np.ones_like(adj_np)

A_corner = torch.FloatTensor(A_corner_np).to(device)
A = torch.FloatTensor(A_np).to(device)
train_mask = torch.FloatTensor(train_mask_np).to(device)
p_sim = torch.FloatTensor(p_sim_np).to(device)
d_sim = torch.FloatTensor(d_sim_np).to(device)

model = ETGPDA(adj_dp, f_dp, A_corner, p_sim, d_sim).to(device)
pred = fit(model, adj, train_mask, lr, num_epochs)
max_allocated_memory = torch.cuda.max_memory_allocated()
print(f"最大已分配内存量: {max_allocated_memory / 1024 ** 2} MB")

pred_np = pred.cpu().detach().numpy()

np.savetxt('pred_np_piRPheno.csv', pred_np, delimiter=",")
