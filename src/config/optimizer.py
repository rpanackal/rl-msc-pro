from pydantic import BaseModel, SerializeAsAny, Field
from pydantic_settings import BaseSettings
from pathlib import PurePath

class SchedulerConfig(BaseSettings):
    name: str = Field("", description="Name of the scheduler configuration")

class CosineAnnealingLRConfig(SchedulerConfig):
    name: str = Field("cosine annealing", description="Specific name of this cosine annealing learning rate scheduler")
    min_lr: float = Field(0.001, description="Minimum learning rate for the cosine annealing schedule")

class OptimizerConfig(BaseSettings):
    name: str = Field("", description="Name of the optimizer configuration")
    lr: float = Field(0.5, description="Learning rate for the optimizer")

    scheduler: SerializeAsAny[SchedulerConfig] | None = Field(None, description="Optional configuration for learning rate scheduler")
