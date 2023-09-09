import torch
from torch import nn
from ..layers import (
    PositionalEncoding,
    PositionWiseFeedForward,
    MultiHeadAttention,
    ContinuousEmbedding,
)
import math


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
        full_output: bool = False,
    ) -> None:
        super().__init__()

        self.prefix_length = math.floor(src_seq_length * cond_prefix_frac)
        self.src_feat_dim = src_feat_dim
        self.tgt_feat_dim = tgt_feat_dim
        self.embed_dim = embed_dim
        self.src_seq_length = src_seq_length
        self.tgt_seq_length = tgt_seq_length
        self.full_output = full_output

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
                EncoderBlock(embed_dim=embed_dim, n_heads=n_heads, expanse_dim=expanse_dim, dropout=dropout)
                for _ in range(n_enc_blocks)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(embed_dim=embed_dim, n_heads=n_heads, expanse_dim=expanse_dim, dropout=dropout)
                for _ in range(n_dec_blocks)
            ]
        )

        self.fc = nn.Linear(in_features=embed_dim, out_features=tgt_feat_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_enc: torch.FloatTensor,
        x_dec: torch.FloatTensor | None = None,
        full_output: bool = False,
        enc_only: bool = False,
    ):
        full_output = full_output or self.full_output

        # Intilialize encoder and decoder inputs
        x_dec = self.decoder_initializer(x_enc, x_dec)
        # src_mask, tgt_mask = self.generate_mask(x_enc, x_dec)
        src_mask = self.create_src_mask(x_enc)
        tgt_mask = self.create_tgt_mask(x_dec)

        # Encoder section
        src_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(x_enc))
        )
        enc_output = src_embedded
        for enc_block in self.encoder_blocks:
            enc_output = enc_block(enc_output, src_mask)

        if enc_only:
            return enc_output

        # Decoder section
        tgt_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(x_dec))
        )
        dec_output = tgt_embedded
        for dec_block in self.decoder_blocks:
            dec_output = dec_block(dec_output, enc_output, src_mask, tgt_mask)

        dec_output = self.fc(dec_output)

        if full_output:
            return (
                dec_output,
                enc_output,
            )

        return dec_output

    def generate_mask(self, x_enc, x_dec):
        """The source mask is applied to mask out padding tokens or other irrelevant
        positions in the source sequence. By masking these positions, the model avoids
        attending to them and focuses only on the actual content of the input.

        The target mask is applied to mask out
            1. future positions in the target sequence during training, which we term
            no peak mask.
            2. padding positions as well as irrelavant position alike source mask, at
            all times.

        Assumes padding and irrelevant sequence elements of value 0.

        Args:
            x_enc (_type_): _description_
                shape: (batch_size, src_seq_length, src_feat_dim)
            x_dec (_type_): _description_
                shape: (batch_size, prefix_length + tgt_seq_length, some_dim)

        Returns:
            _type_: _description_
        """
        # TODO: masks might have to be moved to device
        src_mask = (x_enc != 0).all(dim=-1).unsqueeze(1).unsqueeze(3)
        tgt_mask = (x_dec != 0).all(dim=-1).unsqueeze(1).unsqueeze(3)

        seq_length = x_dec.size(1)
        nopeak_mask = torch.tril(torch.ones((seq_length, seq_length), device=x_dec.device)).to(torch.bool)

        tgt_mask = tgt_mask & nopeak_mask

        return src_mask, tgt_mask

    def create_src_mask(self, src):
        # src shape: (batch_size, src_seq_length, src_feat_dim)
        src_mask = (torch.sum(src, dim=-1) == 0)  # shape: (batch_size, src_seq_length)
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)  # shape: (batch_size, 1, 1, src_seq_length)
        return src_mask

    def create_tgt_mask(self, tgt):
        # tgt shape: (batch_size, tgt_seq_length, some_dim)
        
        # Create a mask to avoid attending to future tokens
        attn_shape = (1, 1, self.tgt_seq_length, self.tgt_seq_length)
        future_mask = torch.triu(torch.ones(attn_shape, device=tgt.device), diagonal=1)  # shape: (1, 1, tgt_len, tgt_len)
        
        # Create a mask to avoid attending to padding tokens
        padding_mask = (torch.sum(tgt, dim=-1) == 0).unsqueeze(1).unsqueeze(2)  # shape: (batch_size, 1, 1, tgt_len)
        
        # Combine the two masks
        tgt_mask = future_mask.to(dtype=torch.bool) | padding_mask  # shape: (batch_size, 1, tgt_len, tgt_len)
        
        return tgt_mask
    
    def decoder_initializer(self, x_enc, x_dec=None):
        """_summary_

        Args:
            x_enc (torch.FloatTensor): Input source sequence to encoder in feature space.
                shape: (batch_size, src_seq_length, src_feat_dim)
            x_dec (torch.FloatTensor): Conditioning input to decoder. Added to seasonal initalizaiton
                along the target sequence length. The feature dimension needs to be less than src_feat_dim.
                shape: (batch_size, tgt_seq_length, some_dim)
        """
        batch_size = x_enc.size(0)

        # mean = torch.mean(x_enc, dim=1, keepdim=True).repeat(1, self.tgt_seq_length, 1)
        conditional = torch.zeros(
            [batch_size, self.tgt_seq_length, self.src_feat_dim],
            device=x_enc.device,
        )

        if x_dec is not None:
            assert x_dec.shape[0:2] == (batch_size, self.tgt_seq_length), ValueError(
                "Conditioning input needs to match batch_size and tgt_seq_length."
            )
            assert x_dec.size(-1) <= self.src_feat_dim, ValueError(
                "Conditioning input feat_dim can be atmost src_feat_dim."
            )

            x_dec_feat_dim = x_dec.size(-1)
            conditional[:, :, -x_dec_feat_dim:] = x_dec

        return conditional
