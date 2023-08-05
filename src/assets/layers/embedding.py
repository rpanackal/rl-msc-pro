import torch
from torch import nn

def compared_version(ver1, ver2):
    """
    :param ver1
    :param ver2
    :return: ver1< = >ver2 False/True
    """
    list1 = str(ver1).split(".")
    list2 = str(ver2).split(".")

    for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
        if int(list1[i]) == int(list2[i]):
            pass
        elif int(list1[i]) < int(list2[i]):
            return -1
        else:
            return 1

    return len(list1) == len(list2) or len(list1) >= len(list2)

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

class TokenEmbedding(nn.Module):
    def __init__(self, feat_dim: int, embed_dim: int):
        super(TokenEmbedding, self).__init__()
        padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
        self.tokenConv = nn.Conv1d(in_channels=feat_dim, out_channels=embed_dim,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)