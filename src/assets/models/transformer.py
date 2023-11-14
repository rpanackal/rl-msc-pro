import torch
from torch import nn
from ..layers import (
    PositionalEncoding,
    PositionWiseFeedForward,
    MultiHeadAttention,
    ContinuousEmbedding,
    AttentionPooling,
)
import math
from pydantic import BaseModel


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, expanse_dim, dropout) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(embed_dim, n_heads)
        self.feed_forward = PositionWiseFeedForward(embed_dim, expanse_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        attn_output = self.self_attn(x, x, x, attn_mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, expanse_dim, dropout) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(embed_dim, n_heads)
        self.cross_attn = MultiHeadAttention(embed_dim, n_heads)
        self.feed_forward = PositionWiseFeedForward(embed_dim, expanse_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_feat_dim,
        tgt_feat_dim,
        embed_dim: int,
        expanse_dim: int,
        n_enc_blocks: int,
        n_dec_blocks: int,
        n_heads: int,
        src_seq_length: int,
        tgt_seq_length: int,
        cond_prefix_frac: float,
        dropout,
        head_dims: list[int] | None = None,
        full_output: bool = False,
    ) -> None:
        super().__init__()

        self.prefix_length = math.floor(src_seq_length * cond_prefix_frac)

        self.encoder_embedding = ContinuousEmbedding(
            feat_dim=src_feat_dim, embed_dim=embed_dim
        )
        self.decoder_embedding = ContinuousEmbedding(
            feat_dim=src_feat_dim, embed_dim=embed_dim
        )
        self.positional_encoding = PositionalEncoding(
            embed_dim=embed_dim, max_seq_length=(src_seq_length + tgt_seq_length)
        )
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    expanse_dim=expanse_dim,
                    dropout=dropout,
                )
                for _ in range(n_enc_blocks)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    expanse_dim=expanse_dim,
                    dropout=dropout,
                )
                for _ in range(n_dec_blocks)
            ]
        )

        self.fc = nn.Linear(in_features=embed_dim, out_features=tgt_feat_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize the additional heads
        if head_dims:
            self.additional_heads = nn.ModuleList(
                [nn.Linear(embed_dim * src_seq_length, dim) for dim in head_dims]
            )

        self.src_feat_dim = src_feat_dim
        self.tgt_feat_dim = tgt_feat_dim
        self.embed_dim = embed_dim
        self.expanse_dim = expanse_dim
        self.n_enc_blocks = n_enc_blocks
        self.n_dec_blocks = n_dec_blocks
        self.n_heads = n_heads
        self.src_seq_length = src_seq_length
        self.tgt_seq_length = tgt_seq_length
        self.cond_prefix_frac = cond_prefix_frac
        self.dropout = dropout
        self.head_dims = head_dims
        self.full_output = full_output

    def forward(
        self,
        source: torch.FloatTensor,
        dec_init: torch.FloatTensor | None = None,
        full_output: bool = False,
        enc_only: bool = False,
    ):
        full_output = full_output or self.full_output

        # Intilialize encoder and decoder inputs
        dec_init = self.decoder_initializer(source, dec_init)

        src_mask = self.create_src_mask(source)
        tgt_mask = self.create_tgt_mask(dec_init)

        # Encoder section
        src_embedded = self.dropout_layer(
            self.positional_encoding(self.encoder_embedding(source))
        )
        enc_output = src_embedded
        for enc_block in self.encoder_blocks:
            enc_output = enc_block(enc_output, src_mask)

        if self.head_dims:
            flattened_enc_output = torch.flatten(
                enc_output, start_dim=1, end_dim=2
            )  # end_dim inclusive
            additional_outputs = [
                head(flattened_enc_output) for head in self.additional_heads
            ]

        if enc_only:
            if self.head_dims:
                return enc_output, additional_outputs
            return enc_output

        # Decoder section
        tgt_embedded = self.dropout_layer(
            self.positional_encoding(self.decoder_embedding(dec_init))
        )
        dec_output = tgt_embedded
        for dec_block in self.decoder_blocks:
            dec_output = dec_block(dec_output, enc_output, src_mask, tgt_mask)

        dec_output = self.fc(dec_output)

        if full_output:
            if self.head_dims:
                return (dec_output, enc_output, additional_outputs)

            return (
                dec_output,
                enc_output,
            )

        return dec_output

    def create_src_mask(self, src):
        # src shape: (batch_size, src_seq_length, src_feat_dim)
        src_mask = torch.sum(src, dim=-1) == 0  # shape: (batch_size, src_seq_length)
        src_mask = src_mask.unsqueeze(1).unsqueeze(
            2
        )  # shape: (batch_size, 1, 1, src_seq_length)
        return src_mask

    def create_tgt_mask(self, tgt):
        # tgt shape: (batch_size, tgt_seq_length, some_dim)

        # Create a mask to avoid attending to future tokens
        attn_shape = (1, 1, self.tgt_seq_length, self.tgt_seq_length)
        future_mask = torch.triu(
            torch.ones(attn_shape, device=tgt.device), diagonal=1
        )  # shape: (1, 1, tgt_len, tgt_len)

        # Create a mask to avoid attending to padding tokens
        padding_mask = (
            (torch.sum(tgt, dim=-1) == 0).unsqueeze(1).unsqueeze(2)
        )  # shape: (batch_size, 1, 1, tgt_len)

        # Combine the two masks
        tgt_mask = (
            future_mask.to(dtype=torch.bool) | padding_mask
        )  # shape: (batch_size, 1, tgt_len, tgt_len)

        return tgt_mask

    def decoder_initializer(self, source, dec_init=None):
        """_summary_

        Args:
            source (torch.FloatTensor): Input source sequence to encoder in feature space.
                shape: (batch_size, src_seq_length, src_feat_dim)
            dec_init (torch.FloatTensor): Conditioning input to decoder. Added to seasonal initalizaiton
                along the target sequence length. The feature dimension needs to be less than src_feat_dim.
                shape: (batch_size, tgt_seq_length, some_dim)
        """
        batch_size = source.size(0)

        mean = torch.mean(source, dim=1, keepdim=True).repeat(1, self.tgt_seq_length, 1)
        # ? padding mask of tgt masking procedure will completely mask the decoder block inputs ?.
        # conditional = torch.zeros(
        #     [batch_size, self.tgt_seq_length, self.src_feat_dim],
        #     device=x_enc.device,
        # )

        if dec_init is not None:
            assert dec_init.shape[0:2] == (batch_size, self.tgt_seq_length), ValueError(
                "Conditioning input needs to match batch_size and tgt_seq_length."
            )
            assert dec_init.size(-1) <= self.src_feat_dim, ValueError(
                "Conditioning input feat_dim can be atmost src_feat_dim."
            )

            dec_init_feat_dim = dec_init.size(-1)
            mean[:, :, -dec_init_feat_dim:] = dec_init

        if self.prefix_length:
            mean = torch.cat([source[:, -self.prefix_length :, :], mean], dim=1)
        return mean

    def model_twin(self):
        """Return a new instance of the same class with the same configuration."""
        return self.__class__(
            src_feat_dim=self.src_feat_dim,
            tgt_feat_dim=self.tgt_feat_dim,
            embed_dim=self.embed_dim,
            expanse_dim=self.expanse_dim,
            n_enc_blocks=self.n_enc_blocks,
            n_dec_blocks=self.n_dec_blocks,
            n_heads=self.n_heads,
            src_seq_length=self.src_seq_length,
            tgt_seq_length=self.tgt_seq_length,
            cond_prefix_frac=self.cond_prefix_frac,
            dropout=self.dropout,
            head_dims=self.head_dims,
            full_output=self.full_output,
        )
