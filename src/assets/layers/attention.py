import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.embed_dim = embed_dim
        self.n_heads = n_heads

        assert embed_dim % n_heads == 0, ValueError(
            "embed_dim must be divisible by n_heads"
        )
        self.head_dim = self.embed_dim // self.n_heads

        # ? Should the in out embedding dim be head_dim?
        # ? Is three different projection used always?
        self.W_q = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.W_k = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.W_v = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.W_o = nn.Linear(self.n_heads * self.head_dim, self.embed_dim, bias=True)

    def forward(self, queries, keys, values, attn_mask=None):
        """Apply multi-head attention mechanism and generate the output.

        Args:
            queries (_type_): _description_
                shape: (batch_size, seq_length, embed_dim)
            keys (_type_): _description_
                shape: (batch_size, seq_length, embed_dim)
            values (_type_): _description_
                shape: (batch_size, seq_length, embed_dim)
            attn_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        Q = self.split_heads(
            self.W_q(queries)
        )  # (batch_size, n_heads, seq_length, head_dim)
        K = self.split_heads(
            self.W_k(keys)
        )  # (batch_size, n_heads, seq_length, head_dim)
        V = self.split_heads(
            self.W_v(values)
        )  # (batch_size, n_heads, seq_length, head_dim)

        attn_scores = self.scaled_dot_product_attention(Q, K, attn_mask)

        # Applying softmax to normalize and produce probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

        multihead_context = (
            attn_probs @ V
        )  # (batch_size, n_heads, seq_length, head_dim)

        return self.W_o(self.combine_heads(multihead_context))

    def split_heads(self, x):
        """Split input vector along embedding dimension and swap seq_length
        and n_heads dimension

        x.shape = (batch_size, seq_length, embed_dim)
          -> (batch_size, seq_length, n_heads, head_dim)
          -> (batch_size, n_heads, seq_length, head_dim)

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
                shape: (batch_size, n_heads, seq_length, head_dim)
        """
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(
            1, 2
        )

    def scaled_dot_product_attention(self, Q, K, attn_mask=None):
        """Calculate attention probabilites and attention weighted
        sum of value vectors.

        Args:
            Q (_type_): _description_
                shape : (batch_size, n_heads, seq_length, head_dim)
            K (_type_): _description_
                shape : (batch_size, n_heads, seq_length, head_dim)
            attn_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
                shape : (batch_size, n_heads, seq_length, seq_length)
        """
        # Divising by square root of keys dimension
        attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # (batch_size, n_heads, seq_length, seq_length)

        # Fill those positions of product matrix as (-1e9) where attn_mask positions are 0
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == True, -1e9)

        return attn_scores

    def combine_heads(self, x):
        """Concatenate output from each head to produce
        a full context vector of length embed_dim

        Args:
            x (_type_):
                shape : (batch_size, n_heads, seq_length, head_dim)

        Returns:
            _type_: _description_
                shape: (batch_size, seq_length, self.embed_dim)
        """
        batch_size, _, seq_length, _ = x.size()
        return (
            x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        )


class AutoCorrelation(nn.Module):
    """AutoCorrelation Mechanism with the following two phases:
        (1) period-based dependencies discovery
        (2) time delay aggregation

    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(
        self, embed_dim: int, n_heads: int, corr_factor: float, *args, **kwargs
    ) -> None:
        """
        Args:
            embed_dim (int): Dimensionality of embedding space of sequence elements.
            n_heads (int): Number of autocorrelation heads.
            corr_factor (float): A hyperparameter that controls number of top
                auto correlation delays considered.

                k = int(corr_factor * math.log(seq_length))

        """
        super().__init__(*args, **kwargs)

        self.embed_dim = embed_dim
        self.n_heads = n_heads

        assert 1 <= corr_factor <= 3, ValueError(
            "Correlation factor not in range [1, 3]"
        )
        self.corr_factor = corr_factor

        assert embed_dim % n_heads == 0, ValueError(
            "embed_dim must be divisible by n_heads"
        )
        self.head_dim = self.embed_dim // self.n_heads

        # ? Should bias be an function parameter ?
        self.W_q = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.W_k = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.W_v = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.W_o = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, queries, keys, values, attn_mask=None):
        """_summary_

        Args:
            queries (_type_): _description_
                shape: (batch_size, seq_length_q, embed_dim)
            keys (_type_): _description_
                shape: (batch_size, seq_length_kv, embed_dim)
            values (_type_): _description_
                shape: (batch_size, seq_length_kv, embed_dim)
            attn_mask (_type_): _description_

        Returns:
            _type_: _description_
                shape: (batch_size, seq_length_q, embed_dim)
        """

        Q = self.split_heads(self.W_q(queries))
        K = self.split_heads(self.W_k(keys))
        V = self.split_heads(self.W_v(values))

        K, V = self.align_lengths(Q, K, V)

        auto_corr_scores = self.auto_correlation(
            Q.permute(0, 2, 3, 1).contiguous(), K.permute(0, 2, 3, 1).contiguous()
        )
        # ? Why are the masks not used?
        multihead_context = self.time_delay_aggregation(
            auto_corr_scores, V.permute(0, 2, 3, 1).contiguous()
        ).permute(0, 3, 1, 2)

        return self.W_o(self.combine_heads(multihead_context))

    def time_delay_aggregation(self, auto_corr_scores, values):
        """Performs time delay aggregation by combining values from different
        time delays weighted by auto correlation scores.

        *Note: Implements the speed up version, according to original paper

        Args:
            auto_corr_scores (_type_): _description_
                shape: (batch_size, n_heads, head_dim, seq_length)
            values (_type_): _description_
                shape: (batch_size, n_heads, head_dim, seq_length)

        Returns:
            _type_: _description_
                shape: (batch_size, n_heads, head_dim, seq_length)
        """
        batch_size, seq_length = values.size(0), values.size(3)

        # Find top k autocorrelations delays
        k = int(self.corr_factor * math.log(seq_length))

        auto_corr_mean = torch.mean(
            auto_corr_scores, dim=(1, 2)
        )  # (batch_size, seq_length)

        if self.training:
            auto_corr_batch_mean = torch.mean(auto_corr_mean, dim=0)
            _, topk_delays = torch.topk(auto_corr_batch_mean, k, dim=-1)  # _, (k)
            # Retrieve top k auto corr values in auto_corr_mean (batch_size, seq_length) using
            # index derived from top k auto corr values in auto_corr_batch_mean (seq_length)
            topk_auto_corr = auto_corr_mean[:, topk_delays]  # (batch_size, k)
        else:
            topk_auto_corr, topk_delays = torch.topk(
                auto_corr_mean, k, dim=-1
            )  # (batch_size, k), (batch_size, k)

        # Applying softmax to normalize and produce probabilities
        topk_auto_corr_probs = torch.softmax(topk_auto_corr, dim=-1)  # (batch_size, k)

        # Compute and aggregate values.roll(delay) * topk_auto_corr_probs(delay)
        # at each top delay values.
        weighted_values_agg = torch.zeros_like(values).float()

        if not self.training:
            # Create an array of indices along sequence length dimension.
            init_indices = (
                torch.arange(seq_length)
                .view(1, 1, 1, -1)
                .repeat(batch_size, self.n_heads, self.head_dim, 1)
                .to(values.device)
            )  # (batch_size, n_heads, head_dim, seq_length)

            # Repeat values along sequence dimension, to apply trick as
            # values_tiled[:, :, :, delay: seq_length+delay] is values.roll(delay)
            values_tiled = values.repeat(
                1, 1, 1, 2
            )  # (batch_size, n_heads, head_dim, seq_length * 2)

        for i in range(k):
            if self.training:
                values_at_delay = values.roll(shifts=-int(topk_delays[i]), dims=-1)
            else:
                # Calculate where each item in sequence would have been without delay
                indices_wo_delay = init_indices + topk_delays[:, i].view(
                    -1, 1, 1, 1
                ).repeat(1, self.n_heads, self.head_dim, seq_length)

                # Value after rolling at each index along sequence length dimension
                # gatherred as auto corr in values_tiled[indices_wo_delay[index]]
                values_at_delay = torch.gather(
                    values_tiled, dim=-1, index=indices_wo_delay
                )  # (batch_size, self.n_heads, self.head_dim, seq_length)

            # For each sample in batch, repeat the i'th normalized auto corr value
            # across all heads, embeddings and sequence elements.
            # Equivalant to broadcasting of topk_auto_corr_probs[:, i].view(-1, 1, 1, 1)
            # with values_at_delay
            topk_auto_corr_at_delay = (
                topk_auto_corr_probs[:, i]
                .view(-1, 1, 1, 1)
                .repeat(1, self.n_heads, self.head_dim, seq_length)
            )

            weighted_values_agg += topk_auto_corr_at_delay * values_at_delay

        # ? Should contiguous be called on weighted_values_agg?
        return weighted_values_agg

    def auto_correlation(
        self, queries: torch.FloatTensor, keys: torch.FloatTensor, dim=1
    ):
        """Compute a measure of similarity between queries and keys in
        the frequency domain using Fast Fourier Transforms based on
        Wiener-Khinchin theorem.

        The series autocorrelation of all lags in {1, · · · , L} is calculated
        at once by FFT. Thus, Auto-Correlation achieves the O(L log L) complexity.

        Args:
            queries (_type_): _description_
                shape: (batch_size, n_heads, head_dim, seq_length)
            keys (_type_): _description_
                shape: (batch_size, n_heads, head_dim, seq_length)

        Returns:
            _type_: Each index in the sequence length dimension corresponds
            to different lag in {1, · · · , L}.
                shape: (batch_size, n_heads, head_dim, seq_length)
        """
        # Transform input to frquency domain
        queries_freq = torch.fft.rfft(queries, dim=-1)
        keys_freq = torch.fft.rfft(keys, dim=-1)

        # Compute correlation scores in frequency domain
        auto_corr_scores_freq = queries_freq * torch.conj(keys_freq)

        seq_length = queries.size(3)
        # Return translated scores from frequency domain back to time domain
        return torch.fft.irfft(auto_corr_scores_freq, n=seq_length, dim=-1)

    def split_heads(self, x: torch.FloatTensor):
        """Split input vector along embedding dimension and swap seq_length
        and n_heads dimension

        x.shape = (batch_size, seq_length, embed_dim)
          -> (batch_size, seq_length, n_heads, head_dim)

        Args:
            x (_type_): _description_
                shape: (batch_size, seq_length, n_heads, head_dim)

        Returns:
            _type_: _description_
        """
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.n_heads, self.head_dim)

    def combine_heads(self, x: torch.FloatTensor):
        """Concatenate output from each head to produce
        a full context vector of length embed_dim

        Args:
            x (_type_):
                shape: (batch_size, seq_length, n_heads, head_dim)

        Returns:
            _type_: _description_
                shape: (batch_size, seq_length, self.embed_dim)
        """
        batch_size, seq_length, _, _ = x.size()
        return x.contiguous().view(batch_size, seq_length, self.embed_dim)

    def align_lengths(self, queries, keys, values):
        """Align the length of queries, keys and values
        to match the query sequence length by zero filling
        or truncation.

        If len(query) < len(key) == len(value), then truncation,
        else If len(query) > len(key) == len(value), zero padding on
            the end of keys and values.

        Args:
            queries (_type_): _description_
                shape: (batch_size, seq_length_q, n_heads, head_dim)
            keys (_type_): _description_
                shape: (batch_size, seq_length_kv, n_heads, head_dim)
            values (_type_): _description_
                shape: (batch_size, seq_length_kv, n_heads, head_dim)

        Returns:
            tuple[torch.FloatTensor, 2]: length aligned keys and values
                0 :
                    shape: (batch_size, seq_length_q, n_heads, head_dim)
                1 :
                    shape: (batch_size, seq_length_q, n_heads, head_dim)
        """
        seq_length_q = queries.size(1)
        seq_length_kv = keys.size(1)
        diff = seq_length_q - seq_length_kv

        if seq_length_q > seq_length_kv:
            zeros = torch.zeros_like(queries[:, :diff, :, :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :seq_length_q, :, :]
            keys = keys[:, :seq_length_q, :, :]

        return keys, values
