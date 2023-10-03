"""
Extending Compatibility wrappers to detangle handling of vector and non-vector environments.
"""
import gymnasium
from gymnasium.logger import deprecation
from gymnasium.utils.step_api_compatibility import step_api_compatibility


class SingleStepAPICompatibility(
    gymnasium.Wrapper, gymnasium.utils.RecordConstructorArgs
):
    """A wrapper which can transform an environment from new step API to old and vice-versa.

    Old step API refers to step() method returning (observation, reward, done, info)
    New step API refers to step() method returning (observation, reward, terminated, truncated, info)
    (Refer to docs for details on the API change)

    Example:
        >>> import gymnasium as gymnasium
        >>> from src.envs.step_api_compatibility import SingleStepAPICompatibility
        >>> env = gymnasium.make("CartPole-v1")
        >>> env # wrapper not applied by default, set to new API
        <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
        >>> env = SingleStepAPICompatibility(gymnasium.make("CartPole-v1"))
        >>> env
        <SingleStepAPICompatibility<TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>>
    """

    def __init__(self, env: gymnasium.Env, output_truncation_bool: bool = True):
        """A wrapper which can transform an environment from new step API to old and vice-versa.

        Args:
            env (gym.Env): the env to wrap. Can be in old or new API
            output_truncation_bool (bool): Whether the wrapper's step method outputs two booleans (new API) or one boolean (old API)
        """
        gymnasium.utils.RecordConstructorArgs.__init__(
            self, output_truncation_bool=output_truncation_bool
        )
        gymnasium.Wrapper.__init__(self, env)

        assert not isinstance(
            env.unwrapped, gymnasium.vector.VectorEnv
        ), "The wrapped environment is vectorized."

        self.output_truncation_bool = output_truncation_bool
        if not self.output_truncation_bool:
            deprecation(
                "Initializing environment in (old) done step API which returns one bool instead of two."
            )

    def step(self, action):
        """Steps through the environment, returning 5 or 4 items depending on `output_truncation_bool`.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info) or (observation, reward, done, info)
        """
        step_returns = self.env.step(action)
        return step_api_compatibility(
            step_returns, self.output_truncation_bool, is_vector_env=False
        )


class VecStepAPICompatibility(
    gymnasium.vector.VectorEnvWrapper, gymnasium.utils.RecordConstructorArgs
):
    """Compatibility wrapper for vector environments based on gymnasium.vector.VectorEnvWrapper."""

    def __init__(self, env: gymnasium.Env, output_truncation_bool: bool = True):
        """A wrapper which can transform an environment from new step API to old and vice-versa.

        Args:
            env (gym.Env): the env to wrap. Can be in old or new API
            output_truncation_bool (bool): Whether the wrapper's step method outputs two booleans (new API) or one boolean (old API)
        """
        gymnasium.utils.RecordConstructorArgs.__init__(
            self, output_truncation_bool=output_truncation_bool
        )
        gymnasium.vector.VectorEnvWrapper.__init__(self, env)

        assert isinstance(
            env.unwrapped, gymnasium.vector.VectorEnv
        ), "The wrapped environment is not vectorized."

        self.output_truncation_bool = output_truncation_bool
        if not self.output_truncation_bool:
            deprecation(
                "Initializing environment in (old) done step API which returns one bool instead of two."
            )

    def step(self, action):
        """Steps through the environment, returning 5 or 4 items depending on `output_truncation_bool`.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info) or (observation, reward, done, info)
        """
        step_returns = self.env.step(action)
        return step_api_compatibility(
            step_returns, self.output_truncation_bool, is_vector_env=True
        )
