from pydantic_settings import BaseSettings
from pydantic import BaseModel, SerializeAsAny
from .optimizer import OptimizerConfig
from .data import BufferConfig
from .assets import VariationalAutoformerConfig, TransformerConfig

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

    buffer: SerializeAsAny[BufferConfig] = BufferConfig()
    actor_optimizer: OptimizerConfig = OptimizerConfig(lr=3e-4)
    critic_optimizer: OptimizerConfig = OptimizerConfig(lr=1e-3)

class CoreticAgentConfig(SACAgentConfig):
    name: str  = "coretic"
    state_seq_length: int = 2
    kappa: float = 0.01

    repr_model: VariationalAutoformerConfig
    repr_model_optimizer: OptimizerConfig = OptimizerConfig(lr=0.1)

class CoretranAgentConfig(SACAgentConfig):
    name: str  = "coretran"
    state_seq_length: int = 2
    kappa: float = 0.01

    repr_model: TransformerConfig
    repr_model_optimizer: OptimizerConfig = OptimizerConfig(lr=0.1)