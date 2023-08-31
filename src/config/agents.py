from pydantic_settings import BaseSettings
from pydantic import BaseModel, SerializeAsAny
from .optimizer import OptimizerConfig
from .data import BufferConfig
from .assets import VariationalAutoformerConfig, AutoformerConfig

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
    log_freq: int = 10

    buffer: BufferConfig = BufferConfig()
    actor_optimizer: OptimizerConfig = OptimizerConfig(lr=3e-4)
    critic_optimizer: OptimizerConfig = OptimizerConfig(lr=1e-3)

class CoreticAgentConfig(SACAgentConfig):
    name: str  = "coretic"
    repr_model: VariationalAutoformerConfig
    repr_model_optimizer: OptimizerConfig = OptimizerConfig(lr=0.1)