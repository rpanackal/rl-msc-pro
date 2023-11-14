from typing import Any, SupportsFloat
import carl
import gym
import gymnasium as gymz
from shimmy.openai_gym_compatibility import _convert_space
from carl.envs.carl_env import CARLEnv
from ...utils import is_vector_env
import numpy as np
from typing import Union

class EnvProjectCompatibility(gymz.Wrapper):
    """A wrapper that adds convenient methods and attributes to non-vectorized environments
    to work well with other environments and utilities.
    """

    def __init__(self, env: Union[gymz.Env, gym.Env]):
        if is_vector_env(env):
            raise ValueError(
                "Use VecEnvProjectCompatibility wrapper for vectorized environments"
            )

        self.is_contextual_env = True if isinstance(env, CARLEnv) else False
        env.unwrapped.metadata["is_contextual_env"] = self.is_contextual_env

        if self.is_contextual_env:
            assert isinstance(
                env.get_wrapper_attr("observation_space"), gymz.spaces.Dict
            ), ValueError("Incompatible contextual environment given.")

        # CARLBraxEnv action space is gym.Space instead of gymz.Space, which is corrected
        # for uniformity
        # if (
        #     self.is_contextual_env
        #     and hasattr(env, "action_space")
        #     and isinstance(env.action_space, gym.Space)
        # ):
        #     env.action_space = _convert_space(env.action_space)

        self.env = env


class VecEnvProjectCompatibility(gymz.vector.VectorEnvWrapper):
    """A wrapper that adds convenient methods and attributes to vector environments
    to work well with other environments and utilities.
    """

    def __init__(self, env: Union[gymz.vector.VectorEnv, gym.vector.VectorEnv]):
        if not is_vector_env(env):
            raise ValueError("The environment given is not vectorized.")

        self.is_contextual_env = env.unwrapped.metadata.get("is_contextual_env", False)
        if self.is_contextual_env:
            assert isinstance(
                env.get_wrapper_attr("single_observation_space"), gymz.spaces.Dict
            ), ValueError("Incompatible contextual environment given.")

        self.env = env
