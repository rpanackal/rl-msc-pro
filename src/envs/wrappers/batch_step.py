from typing import Any, SupportsFloat
from gymnasium.core import Env
import numpy as np
import gymnasium as gymz
from gymnasium.vector import VectorEnvWrapper
from src.utils import is_vector_env


class EnvVectorResponse(gymz.Wrapper):
    """A wrapper that matches non-vector environment response to that of a vector
    environment.
    """
    def __init__(self, env: Env):
        super().__init__(env)

        assert not is_vector_env(env), "The env wrapped must be non-vectorized."
        self.env.single_observation_space = env.observation_space
        self.env.single_action_space = env.action_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        if self.env.is_contextual_env:
            obs['obs'] = obs['obs'][np.newaxis, :]
        else:
            obs = obs[np.newaxis, :]
        return obs, info
    
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        if self.env.is_contextual_env:
            obs['obs'] = obs['obs'][np.newaxis, :]
        else:
            obs = obs[np.newaxis, :]
        
        reward = np.array(reward, dtype=np.float32).reshape(-1)
        terminated = np.array(terminated, dtype=np.bool_).reshape(-1)
        truncated = np.array(truncated, dtype=np.bool_).reshape(-1)
        return obs, reward, terminated, truncated, info