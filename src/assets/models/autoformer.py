import torch
import torch.nn as nn
import math

from ..layers import (
    AutoCorrelation,
    ContinuousEmbedding,
    PositionalEncoding,
    SeriesDecomposition,
)


class FeedForward(nn.Module):
    def __init__(
        self, embed_dim: int, hidden_dim: int, dropout: float, *args, **kwargs
    ) -> None:
        """_summary_

        Args:
            embed_dim (int): Dimensionality of embedding space of sequence elements.
            hidden_dim (int): Dimensionality of space to which embeddings are projected to,
                and then back from.
            dropout (float): _description_
        """
        super().__init__(*args, **kwargs)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(self.dropout(x))


class LayerNorm(nn.Module):
    """Special designed layer normalization for the seasonal part, calculated as:

    LayerNorm(x) = nn.LayerNorm(x) - torch.mean(nn.LayerNorm(x))
    """

    def __init__(self, embed_dim: int, *args, **kwargs) -> None:
        """_summary_

        Args:
            embed_dim (int): Dimensionality of embedding space of sequence elements.
        """
        super().__init__(*args, **kwargs)
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_
                shape: (batch_size, seq_length, embed_dim)

        Returns:
            _type_: _description_
                shape: (batch_size, seq_length, embed_dim)
        """
        x_hat = self.layernorm(x)
        seq_length = x.size(1)

        bias = torch.mean(x_hat, dim=1, keepdim=True).repeat(1, seq_length, 1)
        return x_hat - bias


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        expanse_dim: int,
        kernel_size: int,
        corr_factor: float,
        n_heads: int,
        dropout: float,
        *args,
        **kwargs
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
            n_heads (int): Number of autocorrelation heads.
            dropout (float): _description_
        """
        super().__init__(*args, **kwargs)

        self.auto_corr = AutoCorrelation(
            corr_factor=corr_factor, embed_dim=embed_dim, n_heads=n_heads
        )
        self.series_decomp = SeriesDecomposition(kernel_size=kernel_size)
        self.dropout = nn.Dropout(p=dropout)
        self.ff = FeedForward(
            embed_dim=embed_dim, hidden_dim=expanse_dim, dropout=dropout
        )
        self.layer_norm = LayerNorm(embed_dim=embed_dim)

    def forward(self, x, attn_mask=None):
        """_summary_

        Args:
            x (_type_): _description_
                shape: (batch_size, seq_length, embed_dim)
            attn_mask (_type_): _description_

        Returns:
            _type_: _description_
                shape: (batch_size, seq_length, embed_dim)
        """
        attn_context = self.auto_corr(x, x, x, attn_mask)

        x, _ = self.series_decomp(x + self.dropout(attn_context))

        x, _ = self.series_decomp(x + self.dropout(self.ff(x)))

        return self.layer_norm(x)


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        expanse_dim: int,
        kernel_size: int,
        corr_factor: float,
        n_blocks: int,
        n_heads: int,
        dropout: float,
        *args,
        **kwargs
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
        super().__init__(*args, **kwargs)

        self.enc_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim=embed_dim,
                    expanse_dim=expanse_dim,
                    kernel_size=kernel_size,
                    corr_factor=corr_factor,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x, attn_mask=None):
        """_summary_

        Args:
            x (_type_): _description_
                shape: (batch_size, seq_length, embed_dim)
            attn_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            (Tensor): _description_
                shape: (batch_size, seq_length, embed_dim)

        """
        for enc_block in self.enc_blocks:
            x = enc_block(x, attn_mask)

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        feat_dim: int,
        expanse_dim: int,
        kernel_size: int,
        corr_factor: float,
        n_heads: int,
        dropout: float,
        *args,
        **kwargs
    ) -> None:
        """_summary_

        Args:
            embed_dim (int): Dimensionality of embedding space of sequence elements.
            feat_dim (int): Dimensionality of space to which trend components are finally
                projected to by decoder layer.
            expanse_dim (int): Dimensionality of space to which embeddings are projected to,
                then back from within feed forward layer.
            kernel_size (int): size of the window used for the average pooling to
                compute the trend component.
            corr_factor (float): A hyperparameter that controls number of top
                auto correlation delays considered.
            n_heads (int): Number of autocorrelation heads.
            dropout (float): _description_
        """
        super().__init__(*args, **kwargs)

        self.self_auto_corr = AutoCorrelation(
            corr_factor=corr_factor, embed_dim=embed_dim, n_heads=n_heads
        )
        self.cross_auto_corr = AutoCorrelation(
            corr_factor=corr_factor, embed_dim=embed_dim, n_heads=n_heads
        )
        self.series_decomp = SeriesDecomposition(kernel_size=kernel_size)
        self.ff = FeedForward(
            embed_dim=embed_dim, hidden_dim=expanse_dim, dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)
        self.trend_projection = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=feat_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x, enc_output, cross_mask=None, tgt_mask=None):
        """_summary_

        Args:
            x (_type_): _description_
                shape: (batch_size, prefix_length + tgt_seq_length, embed_dim)
            enc_output (_type_): _description_
                shape: (batch_size, src_seq_length, embed_dim)
            cross_mask (_type_, optional): _description_. Defaults to None.
            tgt_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            tuple[Tensor, 2]
                0: (Tensor): The seasonal part
                    shape: (batch_size, pefex_length + tgt_seq_length, embed_dim)
                1: (Tensor): The trend part
                    shape: (batch_size, prefix_length + tgt_seq_length, tgt_feat_dim)
        """
        attn_context = self.self_auto_corr(x, x, x, tgt_mask)
        x, trend1 = self.series_decomp(x + self.dropout(attn_context))

        attn_context = self.cross_auto_corr(x, enc_output, enc_output, cross_mask)
        x, trend2 = self.series_decomp(x + self.dropout(attn_context))

        x, trend3 = self.series_decomp(x + self.dropout(self.ff(x)))
        # ? Should layer norm be postponed?
        x = self.layer_norm(x)

        block_trend = (trend1 + trend2 + trend3).permute(0, 2, 1)
        block_trend = self.trend_projection(block_trend).permute(0, 2, 1)

        return x, block_trend


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        src_feat_dim: int,
        tgt_feat_dim: int,
        expanse_dim: int,
        kernel_size: int,
        corr_factor: float,
        n_blocks: int,
        n_heads: int,
        dropout: float,
        *args,
        **kwargs
    ) -> None:
        """_summary_

        Args:
            embed_dim (int): Dimensionality of embedding space of sequence elements.
            expanse_dim (int): Dimensionality of space to which embeddings are projected to,
                then back from within feed forward layer.
            feat_dim (int): Dimensionality of space to which trend and seasonal components are
                finally projected to by decoder.
            kernel_size (int): size of the window used for the average pooling to
                compute the trend component.
            corr_factor (float): A hyperparameter that controls number of top
                auto correlation delays considered.
            n_blocks (int): Number of decoder blocks.
            n_heads (int): Number of autocorrelation heads.
            dropout (float): _description_
        """
        super().__init__(*args, **kwargs)

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim=embed_dim,
                    feat_dim=tgt_feat_dim,
                    expanse_dim=expanse_dim,
                    kernel_size=kernel_size,
                    corr_factor=corr_factor,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.seasonal_projection = nn.Linear(embed_dim, tgt_feat_dim, bias=True)
        self.res_trend_projection = nn.Conv1d(
            in_channels=src_feat_dim,
            out_channels=tgt_feat_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False,
        )

    def forward(self, seasonal_init, trend_init, enc_output, cross_mask=None, tgt_mask=None):
        """_summary_

        Args:
            x_seasonal (_type_): _description_
                shape: (batch_size, pefex_length + tgt_seq_length, embed_dim)
            x_trend (_type_): _description_
                shape: (batch_size, prefix_length + tgt_seq_length, embed_dim)
            enc_output (_type_): _description_
                shape: (batch_size, src_seq_length, embed_dim)
            cross_mask (_type_, optional): _description_. Defaults to None.
            tgt_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            tuple[Tensor, 2]
                0: (Tensor): The seasonal part.
                    shape: (batch_size, pefex_length + tgt_seq_length, tgt_feat_dim)
                1: (Tensor): The trend part.
                    shape: (batch_size, prefix_length + tgt_seq_length, tgt_feat_dim)
        """
        trend_residual = self.res_trend_projection(trend_init.permute(0, 2, 1)).permute(
            0, 2, 1
        )
        x_seasonal = seasonal_init

        for dec_block in self.decoder_blocks:
            x_seasonal, trend = dec_block(x_seasonal, enc_output, cross_mask, tgt_mask)

            trend_residual += trend

        trend_final = trend_residual
        return self.seasonal_projection(x_seasonal), trend_final


class Autoformer(nn.Module):
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
        *args,
        **kwargs
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
        """
        super().__init__(*args, **kwargs)

        assert 0 <= cond_prefix_frac <= 1, ValueError(
            "Conditional prefix fraction must be in range [0,1]"
        )
        self.prefix_length = math.floor(src_seq_length * cond_prefix_frac)
        self.src_feat_dim = src_feat_dim
        self.tgt_feat_dim = tgt_feat_dim
        self.embed_dim = embed_dim
        self.src_seq_length = src_seq_length
        self.tgt_seq_length = tgt_seq_length

        self.enc_embedding = ContinuousEmbedding(
            feat_dim=src_feat_dim, embed_dim=embed_dim
        )
        self.dec_embedding = ContinuousEmbedding(
            feat_dim=src_feat_dim, embed_dim=embed_dim
        )
        # ? Token embeddings instead?
        self.positional_encoding = PositionalEncoding(
            embed_dim=embed_dim, max_seq_length=(src_seq_length + tgt_seq_length)
        )
        self.series_decomp = SeriesDecomposition(kernel_size=kernel_size)

        self.encoder = Encoder(
            embed_dim=embed_dim,
            expanse_dim=expanse_dim,
            kernel_size=kernel_size,
            corr_factor=corr_factor,
            n_blocks=n_enc_blocks,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.decoder = Decoder(
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
        x_enc: torch.FloatTensor,
        src_mask=None,
        cross_mask=None,
        tgt_mask=None,
        full_output: bool = False,
        enc_only: bool = False,
    ):
        """_summary_

        Args:
            x_enc (torch.FloatTensor): _description_
                shape: (batch_size, src_seq_length, feat_dim)
            src_mask (_type_, optional): _description_. Defaults to None.
            cross_mask (_type_, optional): _description_. Defaults to None.
            tgt_mask (_type_, optional): _description_. Defaults to None.
            full_output (bool, optional): Whether to output encoder's results. Defaults to False.
            enc_only (bool, optional): Whether to use encoder only. Defaults to False.

        Returns:
            tuple[Tensor, 1 | 2]:
                0: (torch.FloatTensor): The sum of trend and seasonal part produced
                    by decoder.
                    shape: (batch_size, tgt_seq_length, tgt_feat_dim)
                1: (torch.FloatTensor): The encoder's output
                    shape: (batch_size, src_seq_length, embed_dim)
        """
        # initilialization section
        seasonal_init, trend_init = self.decoder_initializer(x_enc)

        # encoder section
        enc_output = self.encoder(
            self.positional_encoding(self.enc_embedding(x_enc)), attn_mask=src_mask
        )

        if enc_only:
            return enc_output

        # decoder section
        seasonal_out, trend_out = self.decoder(
            x_seasonal=self.positional_encoding(self.dec_embedding(seasonal_init)),
            x_trend=trend_init,
            enc_output=enc_output,
            cross_mask=cross_mask,
            tgt_mask=tgt_mask,
        )

        dec_output = (trend_out + seasonal_out)[:, -self.tgt_seq_length :, :]

        if full_output:
            return (
                dec_output,
                enc_output,
            )

        return dec_output

    def decoder_initializer(self, x_enc):
        """_summary_

        Args:
            x_enc (_type_):
                shape: (batch_size, src_seq_length, feat_dim)
        Returns:
            tuple[Tensor, 2]
                0: (_type_): The seasonal initialization
                    shape: (batch_size, pefex_length + tgt_seq_length, embed_dim)
                1: (_type_): The trend initialization.
                    shape: (batch_size, prefix_length + tgt_seq_length, embed_dim)
        """

        batch_size = x_enc.size(0)

        mean = torch.mean(x_enc, dim=1, keepdim=True).repeat(1, self.tgt_seq_length, 1)
        zeros = torch.zeros(
            [batch_size, self.tgt_seq_length, self.src_feat_dim], device=x_enc.device
        )
        seasonal_init, trend_init = self.series_decomp(x_enc)

        seasonal_init = torch.cat(
            [seasonal_init[:, -self.prefix_length :, :], zeros], dim=1
        )
        trend_init = torch.cat([trend_init[:, -self.prefix_length :, :], mean], dim=1)

        return seasonal_init, trend_init
