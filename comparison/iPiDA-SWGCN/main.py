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

weight_adj_path = r"..\data\weight_adj"

adj_df = pd.read_csv(r"..\data\adj.csv", index_col=0)
adj_np = adj_df.values
num_p, num_d = adj_np.shape

p_sim_smith = pd.read_csv(r"..\data\p2p_smith.csv", index_col=0).values

p_feat = rwr(p_sim_smith, 0.8)
np.save(r"..\data\p_feat_smith.npy", p_feat)
p_feat = np.load(r"..\data\p_feat_smith.npy")
d_sim_do = pd.read_csv(r"..\data\d2d_do.csv", index_col=0).values

d_feat1 = rwr(d_sim_do, 0.7)
poly_dis = PolynomialFeatures(3, include_bias=False)
d_feat = poly_dis.fit_transform(d_feat1)


p_sim = torch.FloatTensor(p_sim_smith).to(device)
d_sim = torch.FloatTensor(d_sim_do).to(device)
adj = torch.FloatTensor(adj_np).to(device)
p_feat = torch.FloatTensor(p_feat).to(device)
d_feat = torch.FloatTensor(d_feat).to(device)

lr = 0.001
weight_decay = 0.0
num_epochs = 200

feat_init_d = d_feat.shape[1]


class MaskedMSELoss(nn.MSELoss):
    def forward(self, pred, adj, train_mask, test_mask):
        self.reduction = "none"
        train_loss = (
            super(MaskedMSELoss, self).forward(pred, adj * train_mask * 7) * train_mask
        ).sum()
        test_loss = (super(MaskedMSELoss, self).forward(pred, adj) * test_mask).sum()
        return train_loss, test_loss


def fit(fold_cnt, model, adj, A, train_mask, test_mask, lr, num_epochs):
    optimizer = torch.optim.RMSprop(model.parameters(), lr, weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = MaskedMSELoss()

    test_idx = torch.argwhere(test_mask == 1)
    # test_idx = torch.argwhere(torch.ones_like(test_mask) == 1)
    max_auc_pred = 0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(p_feat, d_feat, p_sim, d_sim, A)
        train_loss, test_loss = loss(pred, adj, train_mask, test_mask)
        train_loss.backward()
        optimizer.step()

        model.eval()
        pred = model(p_feat, d_feat, p_sim, d_sim, A)
        scores = pred[tuple(list(test_idx.T))].cpu().detach().numpy()
        np.save(rf".\scores\f{fold_cnt}_e{epoch}_scores.npy", scores)

        logger.update(
            fold_cnt, epoch, adj, pred, test_idx, train_loss.item(), test_loss.item()
        )

    return pred


with open(r"../data/fold_info.pickle", "rb") as f:
    fold_info = pickle.load(f)

pos_train_ij_list = fold_info["pos_train_ij_list"]
pos_test_ij_list = fold_info["pos_test_ij_list"]
unlabelled_train_ij_list = fold_info["unlabelled_train_ij_list"]
unlabelled_test_ij_list = fold_info["unlabelled_test_ij_list"]
p_gip_list = fold_info["p_gip_list"]
d_gip_list = fold_info["d_gip_list"]
logger = Logger(5)

for i in range(5):
    print(f"fold {i}")
    weight_adj_df = pd.read_csv(rf"{weight_adj_path}\weight_adj_{i}.csv", index_col=0)
    weight_adj_np = weight_adj_df.values

    pos_train_ij = pos_train_ij_list[i]
    pos_test_ij = pos_test_ij_list[i]
    unlabelled_train_ij = unlabelled_train_ij_list[i]
    unlabelled_test_ij = unlabelled_test_ij_list[i]

    A_corner_np = np.zeros_like(weight_adj_df)
    A_corner_np[tuple(list(pos_train_ij.T))] = 1
    A_np = np.concatenate(
        (
            np.concatenate((np.eye(num_p), A_corner_np), axis=1),
            np.concatenate(((A_corner_np).T, np.eye(num_d)), axis=1),
        ),
        axis=0,
    )

    # train_mask_np = np.ones_like(adj_np)
    train_mask_np = np.zeros_like(adj_np)
    train_mask_np[tuple(list(pos_train_ij.T))] = 1
    train_mask_np[tuple(list(unlabelled_train_ij.T))] = 1

    test_mask_np = np.zeros_like(adj_np)
    test_mask_np[tuple(list(pos_test_ij.T))] = 1
    test_mask_np[tuple(list(unlabelled_test_ij.T))] = 1

    A_corner = torch.FloatTensor(A_corner_np).to(device)
    A = torch.FloatTensor(A_np).to(device)
    train_mask = torch.FloatTensor(train_mask_np).to(device)
    test_mask = torch.FloatTensor(test_mask_np).to(device)

    model = GCMC(p_feat, d_feat).to(device)
    pred = fit(i, model, adj, A, train_mask, test_mask, lr, num_epochs)
    max_allocated_memory = torch.cuda.max_memory_allocated()
    print(f"最大已分配内存量: {max_allocated_memory / 1024 ** 2} MB")
logger.save("SWGCN")
