import numpy as np
from utils import *
import torch
from model import *
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib
import pickle

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

seed_everything(42)
device = torch.device("cuda")

import os

path = "scores/"
if not os.path.exists(path):
    os.makedirs(path)

# load adj, sim
adj_df = pd.read_csv(r"..\data\adj.csv", index_col=0)
adj_np = adj_df.values
p_sim_np = pd.read_csv(r"..\data\p2p_smith.csv", index_col=0).values
gensim_feat = np.load(
    r"..\data\gensim_feat_128.npy",
    allow_pickle=True,
).flat[0]
p_kmers_emb = gensim_feat["p_kmers_emb"]
pad_kmers_id_seq = gensim_feat["pad_kmers_id_seq"]

d_sim_np = pd.read_csv(r"..\data\d2d_do.csv", index_col=0).values
d_feat = d_sim_np

num_p, num_d = adj_np.shape


p_sim = torch.FloatTensor(p_sim_np).to(device)
d_sim = torch.FloatTensor(d_sim_np).to(device)
adj = torch.FloatTensor(adj_np).to(device)
p_kmers_emb = torch.FloatTensor(p_kmers_emb).to(device)
pad_kmers_id_seq = torch.tensor(pad_kmers_id_seq).to(device)
d_feat = torch.FloatTensor(d_feat).to(device)

k = 1
merge_win_size = 32
context_size_list = [1, 3, 5]
dll_out_size = 128 * len(context_size_list) * k

gcn_out_dim = 256 * k
gcn_hidden_dim = 256 * k
num_layers, dropout = 1, 0.4

query_size = key_size = 256 * k
value_size = 256 * k
enc_ffn_num_hiddens, n_enc_heads = 256, 2 * k

lr, num_epochs = 0.001, 200

feat_init_d = d_feat.shape[1]


class MaskedBCELoss(nn.BCELoss):
    def forward(self, pred, adj, train_mask):
        self.reduction = "none"
        unweighted_loss = super(MaskedBCELoss, self).forward(pred, adj)
        train_loss = (unweighted_loss * train_mask).sum()
        return train_loss


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def fit(
    model,
    adj,
    adj_full,
    pad_kmers_id_seq,
    d_feat,
    train_mask,
    lr,
    num_epochs,
):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    model.apply(xavier_init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = MaskedBCELoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(pad_kmers_id_seq, d_feat, adj_full)
        train_loss = loss(pred, adj, train_mask)
        train_loss.backward()
        grad_clipping(model, 1)
        optimizer.step()
        print(epoch, train_loss.item())

        # scores = pred[tuple(list(test_idx.T))].cpu().detach().numpy()
        # logger.update(
        #     0, epoch, adj, pred, test_idx, train_loss.item(), train_loss.item()
        # )

    model.eval()
    pred = model(pad_kmers_id_seq, d_feat, adj_full)
    return pred


# logger = Logger(5)

with open(r"../data/p_gip.pickle", "rb") as f:
    p_gip = pickle.load(f)
with open(r"../data/d_gip.pickle", "rb") as f:
    d_gip = pickle.load(f)
with open(f"rn_ij.pickle", "rb") as f:
    rn_ij = pickle.load(f)


pos_ij = np.argwhere(adj_np == 1)
unlabelled_ij = np.argwhere(adj_np == 0)

A_corner_np = np.zeros_like(adj_np)
A_corner_np[tuple(list(pos_ij.T))] = 1

A_np = np.concatenate(
    (
        np.concatenate(((p_sim_np + p_gip) / 2, A_corner_np), axis=1),
        np.concatenate(((A_corner_np).T, (d_sim_np + d_gip) / 2), axis=1),
    ),
    axis=0,
)

train_mask_np = np.zeros_like(adj_np)
train_mask_np[tuple(list(pos_ij.T))] = 1
train_mask_np[tuple(list(rn_ij.T))] = 1


A_corner = torch.FloatTensor(A_corner_np).to(device)
A = torch.FloatTensor(A_np).to(device)
train_mask = torch.FloatTensor(train_mask_np).to(device)

torch.cuda.empty_cache()
deep_lnc_loc = DeepLncLoc(
    p_kmers_emb, dropout, merge_win_size, context_size_list, dll_out_size
).to(device)

gcn = GCN(
    p_feat_dim=dll_out_size,
    d_feat_dim=feat_init_d,
    n_hidden=gcn_hidden_dim,
    dropout=dropout,
).to(device)

p_encoder = TransformerEncoder(
    q_in_dim=gcn_out_dim,
    kv_in_dim=gcn_out_dim,
    key_size=key_size,
    query_size=query_size,
    value_size=value_size,
    ffn_num_hiddens=enc_ffn_num_hiddens,
    num_heads=n_enc_heads,
    num_layers=num_layers,
    dropout=dropout,
    bias=False,
).to(device)

d_encoder = TransformerEncoder(
    q_in_dim=gcn_out_dim,
    kv_in_dim=gcn_out_dim,
    key_size=key_size,
    query_size=query_size,
    value_size=value_size,
    ffn_num_hiddens=enc_ffn_num_hiddens,
    num_heads=n_enc_heads,
    num_layers=num_layers,
    dropout=dropout,
    bias=False,
).to(device)
# predictor = nn.Sequential(
#     nn.Dropout(dropout),
#     nn.Linear(gcn_out_dim, num_d),
#     nn.Sigmoid(),
# ).to(device)
predictor = Predictor().to(device)
model = PDformer(deep_lnc_loc, gcn, p_encoder, d_encoder, predictor).to(device)
pred = fit(
    model,
    adj,
    A,
    pad_kmers_id_seq,
    d_feat,
    train_mask,
    lr,
    num_epochs,
)
scores = pred[tuple(list(unlabelled_ij.T))].cpu().detach().numpy()

pred_np = pred.cpu().detach().numpy()
p_sim_pred = pred_np@(pred_np.T)
d_sim_pred = (pred_np.T)@pred_np
np.savetxt('p_sim_pred.csv', p_sim_pred, delimiter=",")
np.savetxt('d_sim_pred.csv', d_sim_pred, delimiter=",")
np.savetxt('pred_np.csv', pred_np, delimiter=",")

# 创建空的DataFrame
piRNA_name = adj_df.index
disease_name = adj_df.columns
scores_list = []

for i, index in enumerate(unlabelled_ij):
    score = scores[i]
    piRNA = piRNA_name[index[0]]
    disease = disease_name[index[1]]
    scores_list.append([piRNA, disease, score])

scores_df = pd.DataFrame(scores_list, columns=['piRNA', 'disease', 'score'])

scores_df = scores_df.sort_values(by='score', ascending=False)

scores_df.to_csv('unlabelled_scores2.csv', index=False)
