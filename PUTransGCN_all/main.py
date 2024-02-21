import numpy as np
from utils import *
import torch
from model import *
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import pickle
import matplotlib
import os

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

seed_everything(42)
device = torch.device("cuda")
path = "scores/"
if not os.path.exists(path):
    os.makedirs(path)


# load adj, sim
adj_np = pd.read_csv(r"..\data\adj.csv", index_col=0).values
p_sim_np = pd.read_csv(r"..\data\p2p_smith.csv", index_col=0).values
gensim_feat = np.load(
    r"..\data\gensim_feat_128.npy",
    allow_pickle=True,
).flat[0]
p_kmers_emb = gensim_feat["p_kmers_emb"]
pad_kmers_id_seq = gensim_feat["pad_kmers_id_seq"]

d_sim_np = pd.read_csv(r"..\data\d2d_do.csv", index_col=0).values
# d_feat1 = rwr(d_sim, 0.7)
# poly_dis = PolynomialFeatures(3, include_bias=False)
# d_feat = poly_dis.fit_transform(d_feat1)
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

lr, num_epochs = 0.005, 200

feat_init_d = d_feat.shape[1]


class MaskedBCELoss(nn.BCELoss):
    def forward(self, pred, adj, train_mask, test_mask):
        self.reduction = "none"
        unweighted_loss = super(MaskedBCELoss, self).forward(pred, adj)
        train_loss = (unweighted_loss * train_mask).sum()
        test_loss = (unweighted_loss * test_mask).sum()
        return train_loss, test_loss


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
    fold_cnt,
    model,
    adj,
    adj_full,
    pad_kmers_id_seq,
    d_feat,
    train_mask,
    test_mask,
    lr,
    num_epochs,
):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    model.apply(xavier_init_weights)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr, 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = MaskedBCELoss()

    test_idx = torch.argwhere(test_mask == 1)
    # test_idx = torch.argwhere(torch.ones_like(test_mask) == 1)
    # for epoch in range(50):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(pad_kmers_id_seq, d_feat, adj_full)
        train_loss, test_loss = loss(pred, adj, train_mask, test_mask)
        train_loss.backward()
        grad_clipping(model, 1)
        optimizer.step()

        model.eval()
        pred = model(pad_kmers_id_seq, d_feat, adj_full)

        scores = pred[tuple(list(test_idx.T))].cpu().detach().numpy()
        print(len(set(scores)))
        np.save(rf".\scores\f{fold_cnt}_e{epoch}_scores.npy", scores)

        logger.update(
            fold_cnt, epoch, adj, pred, test_idx, train_loss.item(), test_loss.item()
        )
    return 0


logger = Logger(5)

with open(r"../data/fold_info.pickle", "rb") as f:
    fold_info = pickle.load(f)

pos_train_ij_list = fold_info["pos_train_ij_list"]
pos_test_ij_list = fold_info["pos_test_ij_list"]
unlabelled_train_ij_list = fold_info["unlabelled_train_ij_list"]
unlabelled_test_ij_list = fold_info["unlabelled_test_ij_list"]
p_gip_list = fold_info["p_gip_list"]
d_gip_list = fold_info["d_gip_list"]

for i in range(5):
    print(f"fold {i}")
    pos_train_ij = pos_train_ij_list[i]
    pos_test_ij = pos_test_ij_list[i]
    unlabelled_train_ij = unlabelled_train_ij_list[i]
    unlabelled_test_ij = unlabelled_test_ij_list[i]

    p_gip = p_gip_list[i]
    d_gip = d_gip_list[i]

    A_corner_np = np.zeros_like(adj_np)
    A_corner_np[tuple(list(pos_train_ij.T))] = 1

    A_np = np.concatenate(
        (
            np.concatenate(((p_sim_np + p_gip) / 2, A_corner_np), axis=1),
            np.concatenate(((A_corner_np).T, (d_sim_np + d_gip) / 2), axis=1),
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

    model = PUTransGCN(deep_lnc_loc, gcn, p_encoder, d_encoder, predictor).to(device)
    fit(
        i,
        model,
        adj,
        A,
        pad_kmers_id_seq,
        d_feat,
        train_mask,
        test_mask,
        lr,
        num_epochs,
    )
    max_allocated_memory = torch.cuda.max_memory_allocated()
    print(f"最大已分配内存量: {max_allocated_memory / 1024 ** 2} MB")

logger.save("PUTransGCN_all")
