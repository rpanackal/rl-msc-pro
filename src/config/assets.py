from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import torch
from pathlib import PurePath
from .optimizer import OptimizerConfig

class ModelConfig(BaseSettings):
    name: str = Field("", description="Name of the model configuration")

class AutoformerConfig(ModelConfig):
    name: str = Field("autoformer", frozen=True, description="Specific name for the Autoformer model")
    embed_dim: int = Field(256, description="Embedding dimension")
    dropout: float = Field(0.05, description="Dropout rate")
    expanse_dim: int = Field(512, description="Expanse dimension size")
    kernel_size: int = Field(25, description="Size of the kernel")
    corr_factor: float = Field(1, description="Correction factor")
    n_enc_blocks: int = Field(2, description="Number of encoder blocks")
    n_dec_blocks: int = Field(1, description="Number of decoder blocks")
    n_heads: int = Field(8, description="Number of attention heads")
    cond_prefix_frac: float = Field(0.3, description="Fraction of the source that is prefixed to dec_init.")
    src_seq_length: int = Field(50, description="Source sequence length")
    tgt_seq_length: int = Field(50, description="Target sequence length")

class VariationalAutoformerConfig(AutoformerConfig):
    name: str = Field("variational-autoformer", frozen=True, description="Specific name for the Variational Autoformer model")
    kl_weight: float = Field(0.5, description="Weight for the KL divergence")

class TransformerConfig(ModelConfig):
    name: str = Field("transformer", frozen=True, description="Specific name for the Transformer model")

    embed_dim: int = Field(256, description="Embedding dimension")
    expanse_dim: int = Field(512, description="Expanse dimension size")
    n_enc_blocks: int = Field(2, description="Number of encoder blocks")
    n_dec_blocks: int = Field(1, description="Number of decoder blocks")
    n_heads: int = Field(8, description="Number of attention heads")
    src_seq_length: int = Field(50, description="Source sequence length")
    tgt_seq_length: int = Field(50, description="Target sequence length")
    cond_prefix_frac: float = Field(0.3, description="Fraction of the source that is prefixed to dec_init.")
    dropout: float = Field(0.05, description="Dropout rate")
    head_dims: list[int] | None = Field(None, description="Additonal heads on encoder")
    
    load_from_path: str = Field("", description="Path if any to load pretrained model")
class VariationalTransformerConfig(TransformerConfig):
    name: str = Field("variational-transformer", frozen=True, description="Specific name for the Variational Transformer model")
    kl_weight: float = Field(0.5, description="Weight for the KL divergence")

class OrigAutoformerConfig(ModelConfig):
    name: str = Field("orig-autoformer", frozen=True, description="Specific name for the Original Autoformer model")
    output_attention: bool = Field(False, description="Whether the model outputs attention values or not")
    seq_len: int = Field(96, description="Sequence length")
    label_len: int = Field(48, description="Length of the label")
    pred_len: int = Field(96, description="Length of the predictions")
    enc_in: int
    dec_in: int
    c_out: int
    e_layers: int = Field(2, description="Number of encoder layers")
    d_layers: int = Field(1, description="Number of decoder layers")
    d_model: int = Field(512, description="Model's dimension size")
    d_ff: int = Field(2048, description="Feed-forward layer's dimension size")
    embed: str = Field('timeF', description="Type of embedding used")
    freq: int = Field('h', description="Frequency parameter")
    dropout: float = Field(0.05, description="Dropout rate")
    factor: int = Field(1, description="Factor parameter")
    n_heads: int = Field(8, description="Number of attention heads")
    activation: str = Field("gelu", description="Activation function type")
    moving_avg: int = Field(25, description="Moving average parameter")
