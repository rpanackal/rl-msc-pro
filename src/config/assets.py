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
    dropout: float = 0.3
    expanse_dim: int = 2 * embed_dim
    kernel_size: int = 25
    corr_factor: float = 1
    n_enc_blocks: int = 2
    n_dec_blocks: int = 1
    n_heads: int =  8
    cond_prefix_frac: float = 0.3
