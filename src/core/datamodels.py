from pydantic import BaseModel

class LearningEpochResult(BaseModel):
    train_loss: float
    valid_loss: float
    epoch: int
    incumbent_loss: float
    incumbent_found: bool

class EvaluationEpochResult(BaseModel):
    loss: float