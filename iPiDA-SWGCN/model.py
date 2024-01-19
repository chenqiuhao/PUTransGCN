#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import dense, norm

# set random generator seed to allow reproducibility
t.manual_seed(66)


class GCMC(nn.Module):
    def __init__(self, X_pi, X_dis):
        """
                inspired by "HPOFiller: identifying missing protein–phenotype associations by graph convolutional network:
                url:https://github.com/liulizhi1996/HPOFiller
        :param m_pi: number of piRNAs
        :param n_dis: number of diseases
        :param X_pi: m_pi x dim, feature matrix of piRNAs
        :param X_dis: n_term x dim, feature matrix of diseases
        """
        super(GCMC, self).__init__()

        self.bm_input_x = norm.BatchNorm(X_pi.shape[1])
        self.bm_input_y = norm.BatchNorm(X_dis.shape[1])

        self.dense_x = nn.Linear(X_pi.shape[1], 800)
        self.dense_y = nn.Linear(X_dis.shape[1], 800)

        self.bm_x0 = norm.BatchNorm(800)
        self.bm_y0 = norm.BatchNorm(800)

        self.gcn_xy1 = dense.DenseGraphConv(800, 800)
        self.gcn_x1 = dense.DenseGCNConv(800, 800)
        self.gcn_y1 = dense.DenseGCNConv(800, 800)
        self.bm_xy1 = norm.BatchNorm(800)
        self.bm_x1 = norm.BatchNorm(800)
        self.bm_y1 = norm.BatchNorm(800)

        self.gcn_xy2 = dense.DenseGraphConv(800, 800)
        self.gcn_x2 = dense.DenseGCNConv(800, 800)
        self.gcn_y2 = dense.DenseGCNConv(800, 800)
        self.bm_xy2 = norm.BatchNorm(800)
        self.bm_x2 = norm.BatchNorm(800)
        self.bm_y2 = norm.BatchNorm(800)

        self.linear_x1 = nn.Linear(800, 400)
        self.linear_y1 = nn.Linear(800, 400)
        self.linear_x2 = nn.Linear(400, 200)
        self.linear_y2 = nn.Linear(400, 200)
        self.linear_x3 = nn.Linear(200, 100)
        self.linear_y3 = nn.Linear(200, 100)

        self.bm_lx1 = norm.BatchNorm(400)
        self.bm_ly1 = norm.BatchNorm(400)
        self.bm_lx2 = norm.BatchNorm(200)
        self.bm_ly2 = norm.BatchNorm(200)

    def forward(self, X_pi, X_dis, A_pi, A_dis, A_rel):
        """
        :param X_pi: m_pi x dim, feature matrix of piRNAs
        :param X_dis: n_dis x dim, feature matrix of diseases
        :param A_pi: m_pi x m_pi, similarity of piRNAs
        :param A_dis: n_dis x n_dis, similarity of diseases
        :param A_rel: (m_pi + n_dis) x (m_pi + n_dis), piRNA-disease association matrix, i.e. piRNA-disease associations
        :return: predicted piRNA-disease association matrix
        """
        m, n = X_pi.shape[0], X_dis.shape[0]

        X = self.bm_input_x(X_pi)
        Y = self.bm_input_y(X_dis)
        X = F.leaky_relu(self.dense_x(X))
        Y = F.leaky_relu(self.dense_y(Y))
        X = self.bm_x0(X)
        Y = self.bm_y0(Y)

        # Asso-GCN layer1
        XY = t.squeeze(F.leaky_relu(self.gcn_xy1(t.cat((X, Y)), A_rel)))
        XY = self.bm_xy1(XY)
        X, Y = t.split(XY, (m, n))
        # Sim-GCN layer1
        X = t.squeeze(F.leaky_relu(self.gcn_x1(X, A_pi)))
        Y = t.squeeze(F.leaky_relu(self.gcn_y1(Y, A_dis)))
        X = self.bm_x1(X)
        Y = self.bm_y1(Y)

        # Asso-GCN layer2
        XY = t.squeeze(F.leaky_relu(self.gcn_xy2(t.cat((X, Y)), A_rel)))
        XY = self.bm_xy2(XY)
        X, Y = t.split(XY, (m, n))
        # Sim-GCN layer2
        X = t.squeeze(F.leaky_relu(self.gcn_x2(X, A_pi)))
        Y = t.squeeze(F.leaky_relu(self.gcn_y2(Y, A_dis)))
        X = self.bm_x2(X)
        Y = self.bm_y2(Y)

        X = F.relu(self.linear_x1(X))
        Y = F.relu(self.linear_y1(Y))
        X = self.bm_lx1(X)
        Y = self.bm_ly1(Y)
        X = F.relu(self.linear_x2(X))
        Y = F.relu(self.linear_y2(Y))
        X = self.bm_lx2(X)
        Y = self.bm_ly2(Y)
        X = F.relu(self.linear_x3(X))
        Y = F.relu(self.linear_y3(Y))

        return X.mm(Y.t())
