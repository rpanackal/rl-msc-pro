import torch
import torch.nn as nn

from .transformer import Transformer
from ..layers import VariationalLayer
from typing import Union

class VariationalTransformer(Transformer):
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
        super().__init__(
            src_feat_dim,
            tgt_feat_dim=tgt_feat_dim,
            embed_dim=embed_dim,
            expanse_dim=expanse_dim,
            n_enc_blocks=n_enc_blocks,
            n_dec_blocks=n_dec_blocks,
            n_heads=n_heads,
            src_seq_length=src_seq_length,
            tgt_seq_length=tgt_seq_length,
            cond_prefix_frac=cond_prefix_frac,
            dropout=dropout,
            full_output=full_output,
        )

        self.variational_layer = VariationalLayer(embed_dim=embed_dim)


    def forward(
        self,
        x_enc: torch.FloatTensor,
        x_dec: Union[torch.FloatTensor, None] = None,
        full_output: bool = False,
        enc_only: bool = False,
    ):
        full_output = full_output or self.full_output

        # Intilialize decoder inputs
        x_dec = self.decoder_initializer(x_enc, x_dec)

        # src_mask, tgt_mask = self.generate_mask(x_enc, x_dec)
        src_mask = self.create_src_mask(x_enc)
        tgt_mask = self.create_tgt_mask(x_dec) if x_dec is not None else None

        # Encoder section
        src_embedded = self.dropout_layer(
            self.positional_encoding(self.encoder_embedding(x_enc))
        )
        enc_output = src_embedded
        for enc_block in self.encoder_blocks:
            enc_output = enc_block(enc_output, src_mask)

        mean, logvar = self.variational_layer(enc_output)
        latent = self.variational_layer.reparameterize(
            mean, logvar
        )  

        if enc_only:
            return enc_output, mean, logvar, latent

        # Decoder section
        tgt_embedded = self.dropout_layer(
            self.positional_encoding(self.decoder_embedding(x_dec))
        )
        dec_output = tgt_embedded
        for dec_block in self.decoder_blocks:
            dec_output = dec_block(dec_output, enc_output, src_mask, tgt_mask)

        dec_output = self.fc(dec_output)

        if full_output:
            return dec_output, enc_output, mean, logvar, latent

        return dec_output
