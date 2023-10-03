from pydantic import BaseModel, Field, SerializeAsAny
from pydantic_settings import BaseSettings
from .optimizer import OptimizerConfig
from .data import BufferConfig
from .assets import VariationalAutoformerConfig, TransformerConfig, AutoformerConfig

class AgentConfig(BaseSettings):
    name: str = Field("", description="Name of the agent")

class SACAgentConfig(AgentConfig):
    name: str = Field("sac", description="Specific name of the SAC agent")
    gamma: float = Field(0.99, description="Discount factor for the SAC agent")
    tau: float = Field(0.005, description="Soft update coefficient for target network")
    target_network_frequency: int = Field(1, description="Frequency of updating the target network")
    policy_frequency: int = Field(2, description="Frequency of updating the policy")
    alpha: float = Field(0.2, description="Temperature parameter for entropy regularization")
    autotune: bool = Field(True, description="Whether to automatically adjust the temperature parameter alpha")
    noise_clip: float = Field(0.5, description="Clip value for noise (Currently unused)")
    log_freq: int = Field(100, description="Logging frequency")

    buffer: SerializeAsAny[BufferConfig] = Field(BufferConfig(), description="Buffer configuration")
    actor_optimizer: OptimizerConfig = Field(OptimizerConfig(lr=3e-4), description="Optimizer settings for the actor")
    critic_optimizer: OptimizerConfig = Field(OptimizerConfig(lr=1e-3), description="Optimizer settings for the critic")

class CoreticAgentConfig(SACAgentConfig):
    name: str = Field("coretic", description="Specific name for Coretic agent")
    state_seq_length: int = Field(2, description="Sequence length for the agent's state")
    kappa: float = Field(0.01, description="Controls the probability of 0 padding of trajectory sequences.")

    repr_model: VariationalAutoformerConfig | AutoformerConfig
    repr_model_optimizer: OptimizerConfig = Field(OptimizerConfig(lr=0.1), description="Optimizer settings for the representation model")

class CoretranAgentConfig(SACAgentConfig):
    name: str = Field("coretran", description="Specific name for Coretran agent")
    state_seq_length: int = Field(2, description="Sequence length for the agent's state")
    kappa: float = Field(0.01, description="Coefficient value for the agent")

    repr_model: TransformerConfig = Field(TransformerConfig(), description="Configuration for the transformer model used in the agent")
    repr_model_optimizer: OptimizerConfig = Field(OptimizerConfig(lr=0.1), description="Optimizer settings for the representation model")
