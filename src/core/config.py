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
    test_ratio: float = 0.1
    source_ratio: float = 0.5
    crop_length: int = 100
    split_length: int = 100

class D4RLDatasetConfig(DatasetConfig):
    id: str = ""

class DataLoaderConfig(BaseSettings):
    batch_size: int = 32
    shuffle: bool = True

class OptimizerConfig(BaseSettings):
    lr: float = 0.5
    min_lr: float = 0.005

class ExperimentConfig(BaseSettings):
    name: str = ""
    random_seed: int = 42
    n_epochs: int = 20
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset: DatasetConfig = DatasetConfig()
    dataloader: DataLoaderConfig = DataLoaderConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    model: ModelConfig = ModelConfig()
    checkpoint_dir : PurePath = PurePath("src/checkpoints/")


class ReplayBufferConfig(BaseSettings):
    batch_size: int = 64
    buffer_size: int = int(1e6)

class AgentConfig(BaseSettings):
    name: str  = ""

class SACAgentConfig(AgentConfig):
    name: str = "sac"
    gamma: float = 0.99
    tau: float = 0.005
    target_network_frequency: int = 1
    policy_frequency: int = 2
    alpha: float = 0.2
    autotune: bool = True

class RLExperimentConfig(BaseSettings):
    name: str = ""
    env_id: str = "HalfCheetah-v2"
    random_seed: int = 42
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    capture_video: bool = False

    learning_starts: float = 5e3
    total_timesteps: int = 1000000
    noise_clip: float = 0.5

    replay_buffer: ReplayBufferConfig = ReplayBufferConfig()

    actor_optimizer: OptimizerConfig = OptimizerConfig(lr=3e-4)
    critic_optimizer: OptimizerConfig = OptimizerConfig(lr=1e-3)

    agent: AgentConfig = AgentConfig()