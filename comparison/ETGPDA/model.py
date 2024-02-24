#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import dense, norm
import numpy as np


class ETGPDA(nn.Module):
    def __init__(self, adj_dp, f_dp, A_corner, p_sim, d_sim):
        super(ETGPDA, self).__init__()
        device = torch.device("cuda")
        self.num_p, self.num_d = A_corner.shape
        num_total = self.num_p + self.num_d
        num_hidden_f = 16
        self.adj_dp = nn.Dropout(adj_dp)
        self.f_dp = nn.Dropout(f_dp)

        self.h0 = torch.concatenate(
            (
                torch.concatenate((torch.zeros_like(p_sim), A_corner), dim=1),
                torch.concatenate(((A_corner).T, torch.zeros_like(d_sim)), dim=1),
            ),
            dim=0,
        ).to(device)
        adj = torch.concatenate(
            (
                torch.concatenate((6 * p_sim, A_corner), dim=1),
                torch.concatenate(((A_corner).T, 6 * d_sim), dim=1),
            ),
            dim=0,
        )

        row_sum = adj.sum(1)
        degree_mat_inv_sqrt = torch.diag(torch.pow(row_sum, -0.5).flatten())
        self.adj_normalized = (
            adj.mm(degree_mat_inv_sqrt).t().mm(degree_mat_inv_sqrt).to(device)
        )
        self.gcn0 = dense.DenseGCNConv(num_total, num_hidden_f).to(device)
        self.gcn1 = dense.DenseGCNConv(num_hidden_f, num_hidden_f).to(device)
        self.gcn2 = dense.DenseGCNConv(num_hidden_f, num_hidden_f).to(device)
        self.lp0 = nn.Linear(num_hidden_f, num_hidden_f, bias=False).to(device)
        self.lp1 = nn.Linear(num_hidden_f, 32).to(device)
        self.lp2 = nn.Linear(32, 64).to(device)
        self.lp3 = nn.Linear(64, 16).to(device)

    def forward(self):
        adj = self.adj_dp(self.adj_normalized)
        h0_dp = self.f_dp(self.h0)
        h1 = torch.squeeze(F.elu(self.gcn0(h0_dp, adj)))
        h1_dp = self.f_dp(h1)
        h2 = torch.squeeze(F.elu(self.gcn1(h1_dp, adj)))
        h2_dp = self.f_dp(h2)
        h3 = torch.squeeze(F.elu(self.gcn2(h2_dp, adj)))
        emb = 0.5 * h1 + 0.33 * h2 + 0.25 * h3
        emb_dp = self.f_dp(emb)
        R = emb_dp[: self.num_p, :]
        D = emb_dp[self.num_p :, :]
        R1 = self.lp0(R)
        R2 = F.elu(self.lp3(F.elu(self.lp2(F.elu(self.lp1(R))))))
        R3 = (R1 + R2) / 2
        out = F.sigmoid(R3.mm(D.t()))
        return out
