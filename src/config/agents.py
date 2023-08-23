from pydantic_settings import BaseSettings
from pydantic import BaseModel
from .optimizer import OptimizerConfig
from .data import BufferConfig

class AgentConfig(BaseModel):
    name: str  = ""

class SACAgentConfig(AgentConfig):
    name: str = "sac"
    gamma: float = 0.99
    tau: float = 0.005
    target_network_frequency: int = 1
    policy_frequency: int = 2
    alpha: float | None  = 0.2
    autotune: bool = True 
    noise_clip: float = 0.5 # Unused
    
    buffer: BufferConfig = BufferConfig()
    actor_optimizer: OptimizerConfig = OptimizerConfig(lr=3e-4)
    critic_optimizer: OptimizerConfig = OptimizerConfig(lr=1e-3)