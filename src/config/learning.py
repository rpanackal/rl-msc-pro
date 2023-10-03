from pydantic import BaseModel, SerializeAsAny, model_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch
from pathlib import PurePath
from .data import DatasetConfig, DataLoaderConfig
from .optimizer import OptimizerConfig
from .assets import ModelConfig
from .agents import AgentConfig

class SupervisedLearnerConfig(BaseSettings):
    name: str = Field("", description="Name of the supervised learner configuration")
    random_seed: int = Field(42, description="Random seed for reproducibility")
    n_epochs: int = Field(20, description="Number of training epochs")
    device: torch.device = Field(
        default=torch.device("cuda:6" if torch.cuda.is_available() else "cpu"),
        description="Device to run the computations (CPU/CUDA)",
        exclude=True
    )
    checkpoint_dir: PurePath = Field(PurePath("src/checkpoints/"), description="Directory to store model checkpoints")
    
    dataset: SerializeAsAny[DatasetConfig] = Field(default=DatasetConfig(), description="Configuration for dataset")
    dataloader: DataLoaderConfig = Field(default=DataLoaderConfig(), description="Configuration for data loading")
    optimizer: OptimizerConfig = Field(default=OptimizerConfig(), description="Configuration for optimization algorithm")
    model: SerializeAsAny[ModelConfig] = Field(..., description="Configuration for the model")

    model_config : SettingsConfigDict = SettingsConfigDict(arbitrary_types_allowed=True)


class ReinforcedLearnerConfig(BaseSettings):
    name: str = Field("", description="Name of the reinforced learner configuration")
    env_id: str = Field("HalfCheetah-v2", description="ID of the environment")
    random_seed: int = Field(42, description="Random seed for reproducibility")
    device: torch.device = Field(
        default=torch.device("cuda:6" if torch.cuda.is_available() else "cpu"),
        description="Device to run the computations (CPU/CUDA)",
        exclude=True
    )
    capture_video: bool = Field(False, description="Flag to indicate if video should be captured during training")
    checkpoint_dir: PurePath = Field(PurePath("src/checkpoints/"), description="Directory to store model checkpoints")
    
    n_envs: int = Field(1, description="Number of environments for training")
    normalize_observation: bool = Field(True, description="Flag indicating whether to normalize observations or not")
    learning_starts: float = Field(5e3, description="Number of timesteps before training starts")
    total_timesteps: int = Field(1000000, description="Total number of timesteps for training")
    batch_size: int = Field(32, description="Batch size for training")

    agent: SerializeAsAny[AgentConfig] = Field(..., description="Configuration for the agent")

    @model_validator(mode="after")
    def set_name_based_on_env_id(self):
        self.name = self.name or self.env_id
        return self

    model_config : SettingsConfigDict = SettingsConfigDict(arbitrary_types_allowed=True)
