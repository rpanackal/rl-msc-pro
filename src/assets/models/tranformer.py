import torch
from torch import nn
from ..layers import PositionalEncoding, PositionWiseFeedForward, MultiHeadAttention


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, hidden_dim, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.self_attn = MultiHeadAttention(embed_dim, n_heads)
        self.feed_forward = PositionWiseFeedForward(embed_dim, hidden_dim)
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
    def __init__(self, embed_dim, n_heads, hidden_dim, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.self_attn = MultiHeadAttention(embed_dim, n_heads)
        self.cross_attn = MultiHeadAttention(embed_dim, n_heads)
        self.feed_forward = PositionWiseFeedForward(embed_dim, hidden_dim)
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
        src_vocab_size,
        tgt_vocab_size,
        embed_dim,
        n_heads,
        n_blocks,
        hidden_dim,
        max_seq_length,
        dropout,
    ) -> None:

        self.encoder_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(embed_dim, n_heads, hidden_dim, dropout)
                for _ in range(n_blocks)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(embed_dim, n_heads, hidden_dim, dropout)
                for _ in range(n_blocks)
            ]
        )

        self.fc = nn.Linear(embed_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(src))
        )
        tgt_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt))
        )

        enc_output = src_embedded
        for enc_block in self.encoder_blocks:
            enc_output = enc_block(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_block in self.decoder_blocks:
            dec_output = dec_block(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

    def generate_mask(self, src, tgt):
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
            src (_type_): _description_
                shape: (batch_size, seq_length)
            tgt (_type_): _description_
                shape: (batch_size, seq_length)

        Returns:
            _type_: _description_
        """
        # TODO: masks might have to be moved to device
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

        seq_length = tgt.size(1)
        nopeak_mask = torch.tril(torch.ones((seq_length, seq_length))).to(torch.bool)

        tgt_mask = tgt_mask & nopeak_mask

        return src_mask, tgt_mask
