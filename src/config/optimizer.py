from pydantic import BaseModel, SerializeAsAny
from pydantic_settings import BaseSettings
import torch
from pathlib import PurePath

class SchedulerConfig(BaseModel):
    name: str = ""

class CosineAnnealingLRConfig(SchedulerConfig):
    name: str = "cosine annealing"
    min_lr: float = 0.001

class OptimizerConfig(BaseModel):
    name: str = ""
    lr: float = 0.5

    scheduler: SerializeAsAny[SchedulerConfig] | None = None
