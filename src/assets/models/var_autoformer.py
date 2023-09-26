import torch
import torch.nn as nn

from .autoformer import Encoder, Decoder, Autoformer
from ..layers import VariationalLayer


class VariationalEncoder(Encoder):
    """
    Variational Encoder is an extension of the Encoder with an added Variational Layer.
    """

    def __init__(
        self,
        embed_dim: int,
        expanse_dim: int,
        kernel_size: int,
        corr_factor: float,
        n_blocks: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        """_summary_

        Args:
            embed_dim (int): Dimensionality of embedding space of sequence elements.
            expanse_dim (int): Dimensionality of space to which embeddings are projected to,
                then back from within feed forward layer.
            kernel_size (int): size of the window used for the average pooling to
                compute the trend component.
            corr_factor (float): A hyperparameter that controls number of top
                auto correlation delays considered.
            n_blocks (int): Number of encoder blocks.
            n_heads (int): Number of autocorrelation heads.
            dropout (float): _description_
        """
        super().__init__(
            embed_dim=embed_dim,
            expanse_dim=expanse_dim,
            kernel_size=kernel_size,
            corr_factor=corr_factor,
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.variational_layer = VariationalLayer(embed_dim=embed_dim)

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
        mean, logvar = self.variational_layer(x)
        return x, mean, logvar


class VariationalDecoder(Decoder):
    """
    Variational Decoder is an extension of the Decoder.
    It integrates latent variables into the decoder process.
    """

    def forward(
        self,
        seasonal_init,
        trend_init,
        enc_output,
        # latent,
        cross_mask=None,
        tgt_mask=None,
    ):
        """
        Forward pass to decode the encoded representations.

        Args:
            seasonal_init (_type_): _description_
                shape: (batch_size, pefex_length + tgt_seq_length, embed_dim)
            trend_init (_type_): _description_
                shape: (batch_size, prefix_length + tgt_seq_length, src_feat_dim)
            enc_output (Tensor): Output from the Encoder
                shape: (batch_size, src_seq_length, embed_dim)
            latent (Tensor): The latent sampled from latent distribution.
                shape: (batch_size, embed_dim)
            cross_mask (Tensor, optional): Cross-attention mask
            tgt_mask (Tensor, optional): Target attention mask

        Returns:
            tuple[Tensor, 2]
                0: (Tensor): The seasonal part.
                    shape: (batch_size, pefex_length + tgt_seq_length, tgt_feat_dim)
                1: (Tensor): The trend part.
                    shape: (batch_size, prefix_length + tgt_seq_length, tgt_feat_dim)
        """
        # seasonal_init += latent.unsqueeze(1)  # Incorporate latent variable
        # trend_init += latent  # Incorporate latent variable

        return super().forward(
            seasonal_init=seasonal_init,
            trend_init=trend_init,
            enc_output=enc_output,
            cross_mask=cross_mask,
            tgt_mask=tgt_mask,
        )


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
        full_output: bool = False
    ) -> None:
        """_summary_

        Args:
            src_feat_dim (int): Feature dimension target sequence elements
            tgt_feat_dim (int): Feature dimension of target sequence elements
            embed_dim (int): Dimensionality of embedding space of sequence elements.
            expanse_dim (int): Dimensionality of space to which embeddings are projected to,
                then back from within feed forward layer.
            kernel_size (int): size of the window used for the average pooling to
                compute the trend component.
            corr_factor (float): A hyperparameter that controls number of top
                auto correlation delays considered.
            n_enc_blocks (int): Number of encoder blocks.
            n_dec_blocks (int): Number of decoder blocks.
            n_heads (int): Number of autocorrelation heads.
            src_seq_length (int): Length of source sequence
            tgt_seq_length (int): Length of target sequence
            cond_prefix_frac (float): The fraction of the source sequence's ending used
                as a conditional prefix for the decoder input, influencing the target
                prediction.
            dropout (float): _description_
            full_output (bool, optional): Whether to output encoder's results.
                Defaults to False.
        """
        super().__init__(
            src_feat_dim=src_feat_dim,
            tgt_feat_dim=tgt_feat_dim,
            embed_dim=embed_dim,
            expanse_dim=expanse_dim,
            kernel_size=kernel_size,
            corr_factor=corr_factor,
            n_enc_blocks=n_enc_blocks,
            n_dec_blocks=n_dec_blocks,
            n_heads=n_heads,
            src_seq_length=src_seq_length,
            tgt_seq_length=tgt_seq_length,
            cond_prefix_frac=cond_prefix_frac,
            dropout=dropout,
            full_output=full_output
        )

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
        source,
        dec_init: torch.FloatTensor | None = None,
        src_mask=None,
        cross_mask=None,
        tgt_mask=None,
        full_output=False,
        enc_only=False,
    ):
        """
        Forward pass through the Variational Autoformer.

        Args:
            source (torch.FloatTensor): Input source sequence to encoder in feature space.
                shape: (batch_size, src_seq_length, src_feat_dim)
            dec_init (torch.FloatTensor): Conditioning input to decoder. Added to seasonal
                initalizaiton along the target sequence length. The feature dimension needs
                to be less than src_feat_dim.
                shape: (batch_size, tgt_seq_length, some_dim)
            src_mask (_type_, optional): _description_. Defaults to None.
            cross_mask (_type_, optional): _description_. Defaults to None.
            tgt_mask (_type_, optional): _description_. Defaults to None.
            full_output (bool, optional): Whether to output encoder's results.
                Defaults to False.
            enc_only (bool, optional): Whether to use encoder only. Defaults to False.

        Returns:
            dec_output (Tensor): The sum of trend and seasonal part produced
                    by decoder.
                shape: (batch_size, tgt_seq_length, tgt_feat_dim)
            enc_output (Tensor): The Encoder's output.
                shape: (batch_size, src_seq_length, embed_dim)
            mean (Tensor): Mean parameter of the Gaussian .
                shape: (batch_size, src_seq_length, embed_dim)
            logvar (Tensor): Log variance parameter of the Gaussian distribution.
                shape: (batch_size, src_seq_length, embed_dim)
            latent (Tensor): The latent sampled from latent distribution.
                shape: (batch_size, src_seq_length, embed_dim)
        """
        full_output = full_output or self.full_output

        enc_output, mean, logvar = self.encoder(
            self.positional_encoding(self.enc_embedding(source)), src_mask
        )


        latent = self.encoder.variational_layer.reparameterize(
            mean, logvar
        )  # (batch_size, embed_dim)

        if enc_only:
            return enc_output, mean, logvar, latent

        # decoder initilialization section
        seasonal_init, trend_init = self.decoder_initializer(source, dec_init)

        seasonal_out, trend_out = self.decoder(
            seasonal_init=self.positional_encoding(self.dec_embedding(seasonal_init)),
            trend_init=trend_init,
            enc_output=enc_output,
            #latent=latent,
            cross_mask=cross_mask,
            tgt_mask=tgt_mask,
        )

        dec_output = (trend_out + seasonal_out)[:, -self.tgt_seq_length :, :]

        if full_output:
            return dec_output, enc_output, mean, logvar, latent
        return dec_output
