import torch
from torch import nn

class ContinuousEmbedding(nn.Module):
    def __init__(self, feat_dim, embed_dim, bias=False):
        super().__init__()
        self.linear = nn.Linear(feat_dim, embed_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)