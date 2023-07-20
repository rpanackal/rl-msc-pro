import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_length, *args, **kwargs) -> None:
        """For each position in the sequence, the positional encoding vector is calculated by
        combining sine and cosine functions of different frequencies. Each encodimg dimension
        corresponds to a different frequency, capturing different patterns
        and capturing positional information at different scales.

        *Note: Follows the formulation in paper arXiv:1706.03762 [cs.CL]: 'Attention Is All
            You Need'

        Args:
            embed_dim (_type_): _description_
            max_seq_length (_type_): _description_
        """
        super().__init__(*args, **kwargs)

        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_length = x.size(1)
        # ? should requires_grad be false ?
        return x + self.pe[:, :seq_length]
