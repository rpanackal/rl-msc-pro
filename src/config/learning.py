from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings
import torch
from pathlib import PurePath
from .data import DatasetConfig, DataLoaderConfig
from .optimizer import OptimizerConfig
from .assets import ModelConfig
from .agents import AgentConfig

class SupervisedLearnerConfig(BaseModel):
    name: str = ""
    random_seed: int = 42
    n_epochs: int = 20
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_dir : PurePath = PurePath("src/checkpoints/")

    dataset: DatasetConfig = DatasetConfig()
    dataloader: DataLoaderConfig = DataLoaderConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    model: ModelConfig

    class Config:
        arbitrary_types_allowed=True


class ReinforcedLearnerConfig(BaseModel):
    name: str = ""
    env_id: str = "HalfCheetah-v2"
    random_seed: int = 42
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    capture_video: bool = False
    checkpoint_dir : PurePath = PurePath("src/checkpoints/")
    
    learning_starts: float = 5e3
    total_timesteps: int = 1000000
    batch_size: int = 32

    agent: AgentConfig

    @model_validator(mode='after')
    def set_name_based_on_env_id(self):
        self.name = self.name or self.env_id
        return self

    class Config:
        arbitrary_types_allowed=True