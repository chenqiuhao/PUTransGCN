import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class PiRDA(nn.Module):
    def __init__(self, p_onehot2d, d_onehot):
        super(PiRDA, self).__init__()
        self.p_onehot2d = p_onehot2d
        self.d_onehot = d_onehot
        num_d = d_onehot.shape[0]
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=24, kernel_size=7)
        self.gn = nn.GroupNorm(6, 24)
        self.mp = nn.MaxPool1d(2, stride=2)
        self.layer1 = nn.Linear(num_d + 12 * 26, 128)
        self.dropout1 = nn.Dropout(p=0.25)
        self.layer2 = nn.Linear(128, 32)
        self.dropout2 = nn.Dropout(p=0.25)
        self.layer3 = nn.Linear(32, 1)

    def forward(self, ij):
        i = ij[:, 0]
        j = ij[:, 1]

        d_feat = self.d_onehot[j]
        p_feat = self.p_onehot2d[i]
        p_feat_t = p_feat.transpose(2, 1)
        p_conv1d = F.relu(self.conv1d(p_feat_t))  # 24*26
        # p_conv1d = p_conv1d.transpose(2, 1)  # 26*24
        p_gn = self.gn(p_conv1d)  # 24*26
        p_gn_t = p_gn.transpose(2, 1)
        p_feat_flatten = torch.flatten(self.mp(p_gn_t), 1)
        dp_feat = torch.concat((d_feat, p_feat_flatten), dim=1)
        dp_feat1 = self.dropout1(F.relu(self.layer1(dp_feat)))
        dp_feat2 = self.dropout2(F.relu(self.layer2(dp_feat1)))
        score = F.sigmoid(self.layer3(dp_feat2)).flatten()

        return score
