import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import (
    PositionalEncoding,
    AutoCorrelation,
    SeriesDecomposition,
    ContinuousEmbedding,
)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(self.dropout(x))


class LayerNorm(nn.Module):
    """
    Special designed layer normalization for the seasonal part, calculated as:
    LayerNorm(x) = nn.LayerNorm(x) - torch.mean(nn.LayerNorm(x))
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_hat = self.layernorm(x)
        seq_length = x.size(1)

        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, seq_length, 1)
        return x_hat - bias


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        kernel_size,
        corr_factor,
        n_heads,
        dropout,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.auto_corr = AutoCorrelation(corr_factor, embed_dim, n_heads)
        self.series_decomp = SeriesDecomposition(kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.ff = FeedForward(embed_dim, hidden_dim, dropout)
        self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x, attn_mask):
        """_summary_

        Args:
            x (_type_): _description_
                shape: (batch_size, seq_length, embed_dim)
            attn_mask (_type_): _description_
            head_mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        context = self.auto_corr(x, x, x, attn_mask)

        x, _ = self.series_decomp(x + self.dropout(context))

        x, _ = self.series_decomp(x + self.dropout(self.ff(x)))

        return self.layer_norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        trend_dim,
        kernel_size,
        corr_factor,
        n_heads,
        dropout,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.self_auto_corr = AutoCorrelation(corr_factor, embed_dim, n_heads)
        self.cross_auto_corr = AutoCorrelation(corr_factor, embed_dim, n_heads)
        self.series_decomp = SeriesDecomposition(kernel_size)
        self.ff = FeedForward(embed_dim, hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.trend_projection = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=trend_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        context = self.self_auto_corr(x, x, x, tgt_mask)
        x, trend1 = self.series_decomp(x + self.dropout(context))

        context = self.cross_auto_corr(x, enc_output, enc_output, src_mask)
        x, trend2 = self.series_decomp(x + self.dropout(context))

        x, trend3 = self.series_decomp(x + self.dropout(self.ff(x)))

        x = self.layer_norm(x)
        overall_trend = self.trend_projection(
            (trend1 + trend2 + trend3).permute(0, 1, 2)
        ).permute(0, 1, 2)

        return x, overall_trend


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        kernel_size,
        corr_factor,
        n_blocks,
        n_heads,
        dropout,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.enc_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim,
                    hidden_dim,
                    kernel_size,
                    corr_factor,
                    n_heads,
                    dropout,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x, attn_mask):
        for enc_block in self.enc_blocks:
            x = enc_block(x, attn_mask)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        trend_dim,
        kernel_size,
        corr_factor,
        n_blocks,
        n_heads,
        dropout,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.decoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim,
                    hidden_dim,
                    trend_dim,
                    kernel_size,
                    corr_factor,
                    n_heads,
                    dropout,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x, trend_residual, enc_output, src_mask, tgt_mask):
        for dec_block in self.decoder_blocks:
            x, trend = dec_block(x, enc_output, src_mask, tgt_mask)

            trend_residual += trend

        return x


class Autoformer(nn.Module):
    def __init__(
        self,
        src_feat_dim,
        tgt_feat_dim,
        embed_dim,
        hidden_dim,
        trend_dim,
        kernel_size,
        corr_factor,
        n_enc_blocks,
        n_dec_blocks,
        n_heads,
        max_seq_length,
        dropout,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.enc_embedding = ContinuousEmbedding(src_feat_dim, embed_dim)
        self.dec_embedding = ContinuousEmbedding(tgt_feat_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)

        self.encoder = Encoder(
            embed_dim,
            hidden_dim,
            kernel_size,
            corr_factor,
            n_enc_blocks,
            n_heads,
            dropout,
        )

        self.decoder = Decoder(
            embed_dim,
            hidden_dim,
            trend_dim,
            kernel_size,
            corr_factor,
            n_dec_blocks,
            n_heads,
            dropout,
        )

    def forward(self, x):
        pass
