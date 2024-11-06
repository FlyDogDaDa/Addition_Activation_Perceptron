import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, activation=F.relu):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels))
        self.activation = activation

    def forward(self, x):
        # 把x的第一維度unsqueeze，類似於轉置
        x = x.unsqueeze(1)

        # 把x同位相加參數
        x = x + self.weight

        # 把x經過啟動函式
        x = self.activation(x)

        # 把x的通道維度sum在一起
        x = x.sum(dim=2)

        return x
