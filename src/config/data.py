from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

class DatasetConfig(BaseSettings):
    name: str = Field("", description="Name of the dataset configuration")

class D4RLDatasetConfig(DatasetConfig):
    env_id: str = Field(..., description="Environment ID for the dataset")
    validation_ratio: float = Field(0.3, description="Proportion of data set aside for validation")
    test_ratio: float = Field(0.1, description="Proportion of data set aside for testing")
    source_ratio: float = Field(0.5, description="Proportion of data to be used from the source")
    normalize_observation: bool = Field(True, description="Flag indicating whether to normalize observations or not")
    crop_length: int = Field(100, description="Length to crop the dataset sequences")
    split_length: int = Field(100, description="Length to split the dataset sequences into chunks")

    @model_validator(mode='after')
    def set_name_based_on_env_id(self) -> 'D4RLDatasetConfig':
        self.name = self.name or self.env_id
        return self

class DataLoaderConfig(BaseSettings):
    batch_size: int = Field(32, description="Batch size for loading data from the dataset")
    shuffle: bool = Field(True, description="Flag indicating whether to shuffle the dataset while loading or not")

class BufferConfig(BaseSettings):
    buffer_size: int = Field(int(1e5), description="Size of the buffer to store data points")
