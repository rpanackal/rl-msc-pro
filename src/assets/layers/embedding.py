import torch
from torch import nn

class ContinuousEmbedding(nn.Module):
    """Embed any continuous input data, such as numerical features in time series data or 
    sensor measurements.Continuous input data is projected into an embedding space, where 
    each input is represented by a dense vector of fixed dimensions.
    """
    def __init__(self, feat_dim: int, embed_dim: int, bias=False, *args, **kwargs) -> None:
        """_summary_

        Args:
            feat_dim (int): Dimensionality of input feature space.
            embed_dim (int): Dimensionality of embedding space of sequence elements.
            bias (bool, optional): If true, bias is used shift the embeddings based on 
                learned offsets. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(feat_dim, embed_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)