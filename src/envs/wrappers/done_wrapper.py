import numpy as np
import gymnasium

class DoneWrapper(gymnasium.Wrapper):
    """
    A wrapper for gymnasium environments to transform the 'termination' and 'truncation' signals 
    into a single 'done' flag, as used in earlier versions of gym.

    This is particularly useful when interfacing with methods or algorithms that expect the 
    older 'done' convention from gym.

    Attributes:
        is_vector_env (bool): Flag indicating whether the wrapped environment is a vectorized 
                              environment (i.e., processes multiple environments in parallel).
    """

    def __init__(self, env):
        """
        Initialize the DoneWrapper.

        Args:
            env (gymnasium.Env or gymnasium.vector.VectorEnv): The environment to wrap.
        """
        super().__init__(env)
        # Check if the wrapped environment is a vectorized environment
        self.is_vector_env = isinstance(env, gymnasium.vector.VectorEnv)

    def step(self, action):
        """
        Step the environment with the given action. Convert the 'terminated' and 'truncated' 
        signals to the 'done' signal.

        Args:
            action: The action to take in the environment.

        Returns:
            tuple: A tuple containing the observation, reward, done flag, and info dictionary.
        """
        if self.is_vector_env:
            # Handle the case for vectorized environments
            observations, rewards, terminations, truncations, infos = self.env.step(action)
            # Combine 'terminations' and 'truncations' to get the 'done' flag
            dones = np.logical_or(terminations, truncations)
            return observations, rewards, dones, infos
        else:
            # Handle the case for non-vectorized environments
            observation, reward, terminated, truncated, info = self.env.step(action)
            # Combine 'terminated' and 'truncated' to get the 'done' flag
            done = terminated or truncated
            return observation, reward, done, info

    def reset(self, *args, **kwargs):
        """
        Reset the environment to its initial state.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The initial observation of the environment after reset.
        """
        return self.env.reset(*args, **kwargs)
