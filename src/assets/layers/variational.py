import torch
import torch.nn as nn

class VariationalLayer(nn.Module):
    """
    Variational Layer to generate parameters for a Gaussian distribution.
    This layer returns mean and log variance given some input x.
    """

    def __init__(self, embed_dim):
        """
        Initialize the Variational Layer.

        Args:
            embed_dim (int): The dimension of the embedding/hidden layer.
        """
        super().__init__()
        self.mean_layer = nn.Linear(embed_dim, embed_dim)
        self.logvar_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass to compute the mean and log variance.

        Args:
            x (Tensor): Input tensor of shape (batch_size, embed_dim)

        Returns:
            mean (Tensor): Mean parameter of shape (batch_size, embed_dim)
            logvar (Tensor): Log variance parameter of shape (batch_size, embed_dim)
        """
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick to sample from the Gaussian distribution.

        Args:
            mean (Tensor): Mean of the Gaussian distribution.
                shape: (batch_size, embed_dim)
            logvar (Tensor): Log variance of the Gaussian distribution.
                shape: (batch_size, embed_dim)

        Returns:
            z (Tensor): Sampled latent variable from the Gaussian distribution.
                shape: (batch_size, embed_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

