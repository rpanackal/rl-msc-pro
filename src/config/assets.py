from pydantic import BaseModel
from pydantic_settings import BaseSettings
import torch
from pathlib import PurePath
from .optimizer import OptimizerConfig

class ModelConfig(BaseModel):
    name: str = ""

class AutoformerConfig(ModelConfig):
    name: str = "autoformer"
    embed_dim: int = 256
    dropout: float = 0.05
    expanse_dim: int = 512
    kernel_size: int = 25
    corr_factor: float = 1
    n_enc_blocks: int = 2
    n_dec_blocks: int = 1
    n_heads: int =  8
    cond_prefix_frac: float = 0.3
    src_seq_length: int = 50
    tgt_seq_length: int = 50

class VariationalAutoformerConfig(AutoformerConfig):
    name: str = "variational-autoformer"
    
    kl_weight: float = 0.5

class TransformerConfig(ModelConfig):
    name: str = "transformer"

    embed_dim: int = 256
    expanse_dim: int = 512
    n_enc_blocks: int = 2
    n_dec_blocks: int = 1
    n_heads: int = 8
    src_seq_length: int = 50
    tgt_seq_length: int = 50
    cond_prefix_frac: float = 0.3
    dropout: float = 0.05

class VariationalTransformerConfig(TransformerConfig):
    name: str = "variational-transformer"
    
    kl_weight: float = 0.5

class OrigAutoformerConfig(ModelConfig):
    name: str = "orig-autoformer"
    output_attention: bool = False

    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 96
    

    enc_in: int
    dec_in: int
    c_out: int

    e_layers: int = 2
    d_layers: int = 1

    d_model: int = 512
    d_ff: int = 2048
    embed: str = 'timeF'

    freq: int = 'h'
    dropout: float = 0.05

    factor: int = 1
    n_heads: int = 8

    activation: str = "gelu"
    moving_avg: int = 25