import numpy as np
from utils import *
import torch
from model import *
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import warnings
import pickle

warnings.filterwarnings("ignore")
seed_everything(42)
device = torch.device("cuda")


adj_df = pd.read_csv(r"..\data\adj.csv", index_col=0)
adj_np = adj_df.values
num_p, num_d = adj_np.shape

p_sim_smith = pd.read_csv(r"..\data\p2p_smith.csv", index_col=0).values

# p_feat = rwr(p_sim_smith, 0.8)
# np.save(r"..\data\p_feat_smith_rwr.npy", p_feat)
p_feat = np.load(r"..\data\p_feat_smith_rwr.npy")
d_sim_do = pd.read_csv(r"..\data\d2d_do.csv", index_col=0).values

d_feat1 = rwr(d_sim_do, 0.7)
poly_dis = PolynomialFeatures(3, include_bias=False)
d_feat = poly_dis.fit_transform(d_feat1)


p_sim = torch.FloatTensor(p_sim_smith).to(device)
d_sim = torch.FloatTensor(d_sim_do).to(device)
adj = torch.FloatTensor(adj_np).to(device)
p_feat = torch.FloatTensor(p_feat).to(device)
d_feat = torch.FloatTensor(d_feat).to(device)

lr = 0.01
weight_decay = 0.0
num_epochs = 200

feat_init_d = d_feat.shape[1]


class MaskedMSELoss(nn.MSELoss):
    def forward(self, pred, adj, train_mask):
        self.reduction = "none"
        train_loss = (
            super(MaskedMSELoss, self).forward(pred, adj * train_mask * 5) * train_mask
        ).sum()
        return train_loss


def fit(model, adj, A, train_mask, lr, num_epochs):
    # optimizer = torch.optim.RMSprop(model.parameters(), lr, weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = MaskedMSELoss()

    # test_idx = torch.argwhere(torch.ones_like(test_mask) == 1)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(p_feat, d_feat, p_sim, d_sim, A)
        train_loss = loss(pred, adj, train_mask)
        train_loss.backward()
        optimizer.step()

        # model.eval()
        # pred = model(p_feat, d_feat, p_sim, d_sim, A)
        # scores = pred[tuple(list(test_idx.T))].cpu().detach().numpy()
        # np.save(rf".\scores\f{fold_cnt}_e{epoch}_scores.npy", scores)

    model.eval()
    pred = model(p_feat, d_feat, p_sim, d_sim, A)
    return pred


with open(r"../data/p_gip.pickle", "rb") as f:
    p_gip = pickle.load(f)
with open(r"../data/d_gip.pickle", "rb") as f:
    d_gip = pickle.load(f)


pos_ij = np.argwhere(adj_np == 1)
unlabelled_ij = np.argwhere(adj_np == 0)

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
# train_mask_np = np.zeros_like(adj_np)
# train_mask_np[tuple(list(rn_ij.T))] = 1


A_corner = torch.FloatTensor(A_corner_np).to(device)
A = torch.FloatTensor(A_np).to(device)
train_mask = torch.FloatTensor(train_mask_np).to(device)

model = GCMC(p_feat, d_feat).to(device)
pred = fit(model, adj, A, train_mask, lr, num_epochs)
pred_np = pred.cpu().detach().numpy()

max_allocated_memory = torch.cuda.max_memory_allocated()
print(f"最大已分配内存量: {max_allocated_memory / 1024 ** 2} MB")
np.savetxt('pred_np_raw.csv', pred_np, delimiter=",")

