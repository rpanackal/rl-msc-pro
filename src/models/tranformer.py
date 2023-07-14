import torch
from torch import nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, feat_dim, n_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.feat_dim = feat_dim
        self.n_heads = n_heads

        assert feat_dim % n_heads == 0, "feat_dim must be divisible by n_heads"
        self.head_dim = self.feat_dim // self.n_heads

        # ? Should the in out feature dim be head_dim?
        # ? Is three different projection used always?
        self.W_q = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
        self.W_k = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
        self.W_v = nn.Linear(self.feat_dim, self.feat_dim, bias=False)

        self.W_o = nn.Linear(self.n_heads * self.head_dim, self.feat_dim)

    def forward(self, query, key, value, mask=None):
        """Apply multi-head attention mechanism and generate the output.

        Args:
            query (_type_): _description_
                shape: (batch_size, seq_length, feat_dim)
            key (_type_): _description_
                shape: (batch_size, seq_length, feat_dim)
            value (_type_): _description_
                shape: (batch_size, seq_length, feat_dim)
            mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # Arguements key, query and value are of shape
        # (batch_size, seq_length, feat_dim).
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

    def split_heads(self, x):
        """Split input vector along embedding dimension and swap seq_length
        and n_heads dimension

        x.shape = (batch_size, seq_length, feat_dim)
          -> (batch_size, seq_length, n_heads, head_dim)
          -> (batch_size, n_heads, seq_length, head_dim)

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(
            1, 2
        )

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Calculate attention probabilites and attention weighted
        sum of value vectors

        Args:
            Q (_type_): _description_
                shape : (batch_size, n_heads, seq_length, head_dim)
            K (_type_): _description_
                shape : (batch_size, n_heads, seq_length, head_dim)
            V (_type_): _description_
                shape : (batch_size, n_heads, seq_length, head_dim)
            mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
                shape : (batch_size, n_heads, seq_length, head_dim)
        """
        # Divising by square root of key dimension
        attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # (batch_size, n_heads, seq_length, seq_length)

        # Fill those positions of product matrix as (-1e9) where mask positions are 0
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Applying softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)

        return attn_probs @ V

    def combine_heads(self, x):
        """Concatenate output from each head to produce
        a vector of length feat_dim

        Args:
            x (_type_):
                shape : (batch_size, n_heads, seq_length, head_dim)

        Returns:
            _type_: _description_
                shape: (batch_size, seq_length, self.feat_dim)
        """
        batch_size, _, seq_length, _ = x.size()
        return (
            x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.feat_dim)
        )


class PositionWiseFeedForward(nn.Module):
    def __init__(self, feat_dim, d_hidden, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # ? Wh
        self.fc1 = nn.Linear(feat_dim, d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, feat_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, feat_dim, max_seq_length, *args, **kwargs) -> None:
        """For each position in the sequence, the positional encoding vector is calculated by 
        combining sine and cosine functions of different frequencies. Each encodimg dimension
        corresponds to a different frequency, capturing different patterns 
        and capturing positional information at different scales.

        Args:
            feat_dim (_type_): _description_
            max_seq_length (_type_): _description_
        """
        super().__init__(*args, **kwargs)

        pe = torch.zeros(max_seq_length, feat_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feat_dim, 2).float() * -(math.log(10000.0) / feat_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_length = x.size(1)
        #? should requires_grad be false ?
        return x + self.pe[:, :seq_length]


class EncoderBlock(nn.Module):
    def __init__(self, feat_dim, n_heads, hidden_dim, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.self_attn = MultiHeadAttention(feat_dim, n_heads)
        self.feed_forward = PositionWiseFeedForward(feat_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, feat_dim, n_heads, hidden_dim, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.self_attn = MultiHeadAttention(feat_dim, n_heads)
        self.cross_attn = MultiHeadAttention(feat_dim, n_heads)
        self.feed_forward = PositionWiseFeedForward(feat_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.norm3 = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        #? should the encoder output be unpacked ?
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
        feat_dim,
        n_heads,
        num_layers,
        hidden_dim,
        max_seq_length,
        dropout,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.encoder_embedding = nn.Embedding(src_vocab_size, feat_dim)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, feat_dim)
        self.positional_encoding = PositionalEncoding(feat_dim, max_seq_length)

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(feat_dim, n_heads, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(feat_dim, n_heads, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fc = nn.Linear(feat_dim, tgt_vocab_size)
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
        nopeak_mask = torch.tril(torch.ones((seq_length, seq_length))).bool()
        
        tgt_mask = tgt_mask & nopeak_mask

        return src_mask, tgt_mask
