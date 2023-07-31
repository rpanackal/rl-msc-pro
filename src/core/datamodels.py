from pydantic import BaseModel

class EpochResult(BaseModel):
    train_loss: float
    valid_loss: float
    epoch: int
    incumbent_loss: float
    incumbent_found: bool