from pydantic import BaseModel
from pydantic_settings import BaseSettings
import torch
from pathlib import PurePath

class ModelConfig(BaseSettings):
    name: str = ""

class AutoformerConfig(ModelConfig):
    name: str = "autoformer"
    embed_dim: int = 512
    dropout: float = 0.3
    expanse_dim: int = 2 * embed_dim
    kernel_size: int = 25
    corr_factor: float = 1
    n_enc_blocks: int = 2
    n_dec_blocks: int = 1
    n_heads: int =  8
    cond_prefix_frac: float = 0.3

class DatasetConfig(BaseSettings):
    validation_ratio: float = 0.3
    source_ratio: float = 0.5
    crop_length: int = 100

class D4RLDatasetConfig(DatasetConfig):
    id: str = ""

class DataLoaderConfig(BaseSettings):
    batch_size: int = 32
    shuffle: bool = True

class OptimizerConfig(BaseSettings):
    lr: float = 0.0001

class ExperimentConfig(BaseSettings):
    name: str = ""
    random_seed: int = 42
    n_epochs: int = 300
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset: DatasetConfig = DatasetConfig()
    dataloader: DataLoaderConfig = DataLoaderConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    model: ModelConfig = ModelConfig()
    checkpoint_dir : PurePath = PurePath("src/checkpoints/")

setting = ExperimentConfig().model_dump_json(exclude={"device"})