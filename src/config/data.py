from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings
import torch
from pathlib import PurePath


class DatasetConfig(BaseModel):
    name: str = ""

class D4RLDatasetConfig(DatasetConfig):
    env_id: str 
    validation_ratio: float = 0.3
    test_ratio: float = 0.1
    source_ratio: float = 0.5

    crop_length: int = 100
    split_length: int = 100

    @model_validator(mode='after')
    def set_name_based_on_env_id(self) -> 'D4RLDatasetConfig':
        self.name = self.name or self.env_id
        return self

class DataLoaderConfig(BaseModel):
    batch_size: int = 32
    shuffle: bool = True


class BufferConfig(BaseModel):
    buffer_size: int = int(1e5)
    