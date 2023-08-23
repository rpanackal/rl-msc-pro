import torch
import torch.nn as nn

from .autoformer import Encoder, Decoder, Autoformer


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


class VariationalEncoder(Encoder):
    """
    Variational Encoder is an extension of the Encoder with an added Variational Layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variational_layer = VariationalLayer(
            self.enc_blocks[0].auto_corr.embed_dim
        )

    def forward(self, x, attn_mask=None):
        """
        Forward pass to encode the input and compute the mean and log variance.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim)
            attn_mask (Tensor, optional): Attention mask

        Returns:
            x (Tensor): Output tensor of shape (batch_size, seq_length, embed_dim)
            mean (Tensor): Mean parameter of shape (batch_size, embed_dim)
            logvar (Tensor): Log variance parameter of shape (batch_size, embed_dim)
        """
        x = super().forward(x, attn_mask)
        mean, logvar = self.variational_layer(x[:, -1, :])
        return x, mean, logvar


class VariationalDecoder(Decoder):
    """
    Variational Decoder is an extension of the Decoder.
    It integrates latent variables into the decoder process.
    """

    def forward(
        self, x_seasonal, x_trend, latent, enc_output, cross_mask=None, tgt_mask=None
    ):
        """
        Forward pass to decode the encoded representations.

        Args:
            x_seasonal (Tensor): Seasonal component
            x_trend (Tensor): Trend component
            latent (Tensor): Latent variable
            enc_output (Tensor): Output from the Encoder
            cross_mask (Tensor, optional): Cross-attention mask
            tgt_mask (Tensor, optional): Target attention mask

        Returns:
            tuple[Tensor, 2]
                0: (Tensor): The seasonal part.
                    shape: (batch_size, pefex_length + tgt_seq_length, tgt_feat_dim)
                1: (Tensor): The trend part.
                    shape: (batch_size, prefix_length + tgt_seq_length, tgt_feat_dim)
        """
        x_seasonal += latent  # Incorporate latent variables
        x_trend += latent  # Incorporate latent variables
        return super().forward(x_seasonal, x_trend, enc_output, cross_mask, tgt_mask)


class VariationalAutoformer(Autoformer):
    """
    Variational Autoformer combines a Variational Encoder and a Variational Decoder.
    """

    def __init__(
        self,
        src_feat_dim: int,
        tgt_feat_dim: int,
        embed_dim: int,
        expanse_dim: int,
        kernel_size: int,
        corr_factor: float,
        n_enc_blocks: int,
        n_dec_blocks: int,
        n_heads: int,
        src_seq_length: int,
        tgt_seq_length: int,
        cond_prefix_frac: float,
        dropout: float,
    ):
        super().__init__(src_feat_dim,
        tgt_feat_dim,
        embed_dim,
        expanse_dim,
        kernel_size,
        corr_factor,
        n_enc_blocks,
        n_dec_blocks,
        n_heads,
        src_seq_length,
        tgt_seq_length,
        cond_prefix_frac,
        dropout)

        self.encoder = VariationalEncoder(
            embed_dim=embed_dim,
            expanse_dim=expanse_dim,
            kernel_size=kernel_size,
            corr_factor=corr_factor,
            n_blocks=n_enc_blocks,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.decoder = VariationalDecoder(
            embed_dim=embed_dim,
            src_feat_dim=src_feat_dim,
            tgt_feat_dim=tgt_feat_dim,
            expanse_dim=expanse_dim,
            kernel_size=kernel_size,
            corr_factor=corr_factor,
            n_blocks=n_dec_blocks,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(
        self,
        x_enc,
        src_mask=None,
        cross_mask=None,
        tgt_mask=None,
        full_output=False,
        enc_only=False,
    ):
        """
        Forward pass through the Variational Autoformer.

        Args:
            x_enc (Tensor): Encoder input
            src_mask (Tensor, optional): Source mask for the encoder
            cross_mask (Tensor, optional): Cross-attention mask
            tgt_mask (Tensor, optional): Target attention mask
            full_output (bool, optional): Whether to return encoder's output and other 
                intermediaries. Defaults to False.
            enc_only (bool, optional): Whether to only run the encoder, default False

        Returns:
            dec_output (Tensor): The sum of trend and seasonal part produced
                    by decoder.
                shape: (batch_size, tgt_seq_length, tgt_feat_dim)
            enc_output (Tensor): The Encoder's output.
                shape: (batch_size, src_seq_length, embed_dim)
            mean (Tensor): Mean parameter of the Gaussian .
                shape: (batch_size, embed_dim)
            logvar (Tensor): Log variance parameter of the Gaussian distribution.
                shape: (batch_size, embed_dim)
            latent (Tensor): The latent sampled from latent distribution.
                shape: (batch_size, embed_dim)
        """
        enc_output, mean, logvar = self.encoder(x_enc, src_mask)
        latent = self.encoder.variational_layer.reparameterize(mean, logvar)
        
        if enc_only:
            return enc_output, mean, logvar, latent

        seasonal_init, trend_init = self.decoder_initializer(x_enc)
        seasonal_init += latent  # Incorporate latent variable
        trend_init += latent  # Incorporate latent variable

        seasonal_out, trend_out = self.decoder(
            seasonal_init, trend_init, latent, enc_output, cross_mask, tgt_mask
        )

        dec_output =  (trend_out + seasonal_out)[:, -self.tgt_seq_length:, :]
        if full_output:
            return dec_output, enc_output, mean, logvar, latent

        return dec_output
