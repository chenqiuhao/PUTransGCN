#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import dense, norm

# set random generator seed to allow reproducibility
t.manual_seed(42)


class GBNN(nn.Module):
    def __init__(self, num_p, num_d):
        super(GBNN, self).__init__()

        n6 = int(6000 / 10)
        n4 = int(4000 / 10)
        n2 = int(2000 / 10)
        n10 = int(200 / 10)
        self.sae_1 = nn.Linear(num_p, n4)
        self.sae_2 = nn.Linear(n4, n2)
        self.sae_3 = nn.Linear(n2, n10)
        self.sae_4 = nn.Linear(n10, n2)
        self.sae_5 = nn.Linear(n2, n4)
        self.d_dense = nn.Linear(num_d, num_p)

        self.gnet_1 = nn.Linear((n4 + num_d), n6)
        self.gnet_2 = nn.Linear(n6, n2)
        self.gnet_3 = nn.Linear((n4 + num_d + n2), n6)
        self.gnet_4 = nn.Linear(n6, n2)
        self.gnet_5 = nn.Linear((n4 + num_d + n2), n6)
        self.gnet_6 = nn.Linear(n6, n2)

        self.gnet_7 = nn.Linear(n2, num_d)
        self.gnet_8 = nn.Linear(n2, num_d)
        self.gnet_9 = nn.Linear(n2, num_d)

    def forward(self, p_M, d_M):
        p_M1 = self.sae_1(p_M)
        p_M1 = self.sae_2(p_M1)
        p_M1 = self.sae_3(p_M1)
        p_M1 = self.sae_4(p_M1)
        p_M1 = self.sae_5(p_M1)

        d_M1 = self.d_dense(d_M)

        pd_pair = torch.cat((p_M1, d_M1.t()), 1)
        stage1_out = self.gnet_2(self.gnet_1(pd_pair))
        f1x = self.gnet_7(stage1_out)
        gnet_3_in = torch.cat((pd_pair, stage1_out), 1)
        stage2_out = self.gnet_4(self.gnet_3(gnet_3_in))
        f2x = self.gnet_8(stage2_out)
        gnet_5_in = torch.cat((pd_pair, stage2_out), 1)
        stage3_out = self.gnet_6(self.gnet_5(gnet_5_in))
        f3x = self.gnet_9(stage3_out)
        return F.sigmoid((f1x + f2x + f3x) / 3)
