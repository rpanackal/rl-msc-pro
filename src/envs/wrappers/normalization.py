import numpy as np
import gymnasium as gymz
from gymnasium.vector import VectorEnvWrapper
from src.utils import is_vector_env


class RMVNormalizeVecObservation(gymz.Wrapper):
    """
    A wrapper for normalizing observations in a vectorized gymz environment.
    It normalizes observations using a running mean and standard deviation.
    """

    def __init__(
        self,
        env,
        is_observation_scaling=True,
        epsilon=1e-8,
    ):
        """
        Initialize the wrapper by setting the environment to be wrapped
        and initializing mean and variance to None.
        """
        assert is_vector_env(env), "The env wrapped must be vectorized."
        super().__init__(env)

        self.is_contextual_env = env.is_contextual_env

        # Running mean and variance of observations
        if env.is_contextual_env:
            self.mean = np.zeros(
                self.single_observation_space["obs"].shape, dtype=np.float64
            )
            self.var = np.ones(
                self.single_observation_space["obs"].shape, dtype=np.float64
            )
        else:
            self.mean = np.zeros(self.single_observation_space.shape, dtype=np.float64)
            self.var = np.ones(self.single_observation_space.shape, dtype=np.float64)

        self.count = 0  # Count of observations
        self.epsilon = epsilon
        self._is_observation_scaling = is_observation_scaling

    def reset(self, *args, **kwargs):
        """
        Reset the environment and update normalization statistics.
        """
        observations, infos = self.env.reset(*args, **kwargs)

        if self.is_contextual_env:
            self.update_stats(observations["obs"])
            observations["obs"] = (
                self.normalize_observations(observations["obs"])
                if self._is_observation_scaling
                else observations["obs"]
            )
            return observations, infos

        self.update_stats(observations)
        return (
            self.normalize_observations(observations)
            if self._is_observation_scaling
            else observations,
            infos,
        )

    def step(self, actions):
        """
        Step the environment, get new observations,
        and update normalization statistics.
        """
        observations, rewards, terminated, truncated, infos = super().step(actions)

        if self.is_contextual_env:
            self.update_stats(observations["obs"])
            observations["obs"] = (
                self.normalize_observations(observations['obs'])
                if self._is_observation_scaling
                else observations["obs"]
            )
            return observations, rewards, terminated, truncated, infos

        self.update_stats(observations)
        return (
            self.normalize_observations(observations)
            if self._is_observation_scaling
            else observations,
            rewards,
            terminated,
            truncated,
            infos,
        )

    def update_stats(self, observations):
        """
        Update the running mean and standard deviation of observations.

        The method uses Welford's algorithm for computing running variance.
        It is an efficient and numerically stable algorithm.
        """
        # Ensure that observation shape matches self.mean and self.var
        assert observations.shape[1:] == self.mean.shape

        # Compute mean and variance for the current batch of observations
        batch_size = observations.shape[0]
        batch_mean = np.mean(observations, axis=0)
        # Use ddof=1 for unbiased estimator when batch_size > 1
        batch_var = np.var(observations, axis=0, ddof=1 if batch_size > 1 else 0)

        # Compute the new total count of observations
        new_count = self.count + batch_size

        # Compute the delta between the old mean and the new batch mean
        delta = batch_mean - self.mean

        # Update the running mean using the delta and the new count
        new_mean = self.mean + delta * batch_size / new_count

        # Update the running variance using Welford's algorithm
        # n1 is self.count, n2 is batch_size
        # var1 is self.var, var2 is batch_var
        # Add condition new_count > 1, to avoid division by zero error.
        # If it's the first update, set var to batch_var directly.
        if self.count == 0:
            new_var = batch_var
        else:
            new_var = (
                self.var * (self.count - 1)  # (n1 - 1) * var1
                + batch_var * (batch_size - 1)  # (n2 - 1) * var2
                # (mean1 - mean2)^2 * n1 * n2 / (n1 + n2)
                + np.square(delta) * self.count * batch_size / new_count
            ) / (
                new_count - 1
            )  # (n1 + n2 - 1)

        # Update the running statistics for future use
        self.mean, self.var, self.count = new_mean, new_var, new_count

    def normalize_observations(self, observations):
        """
        Normalize observations using the current running mean and standard deviation.
        """
        # Compute the new running standard deviation with a small epsilon for numerical stability
        std = np.sqrt(self.var + self.epsilon)
        return (observations - self.mean) / std


class EMANormalizeVecObservation(VectorEnvWrapper):
    """
    A wrapper for normalizing observations in a vectorized gymz environment.
    It normalizes observations using Exponential Moving Average (EMA).
    """

    def __init__(self, env, alpha=0.99, epsilon=1e-8):
        """
        Initialize the wrapper by setting the environment to be wrapped
        and initializing mean and variance to their initial values.
        """
        super(EMANormalizeVecObservation, self).__init__(env)
        self.alpha = alpha
        self.epsilon = epsilon
        self.mean = np.zeros(self.single_observation_space.shape, dtype=np.float64)
        self.var = np.ones(self.single_observation_space.shape, dtype=np.float64)

    def reset(self):
        """
        Reset the environment and update normalization statistics.
        """
        observations = self.env.reset()
        self.update_stats(observations)
        return self.normalize_observations(observations)

    def step(self, actions):
        """
        Step the environment, get new observations,
        and update normalization statistics.
        """
        observations, rewards, dones, infos = super().step(actions)
        self.update_stats(observations)
        return self.normalize_observations(observations), rewards, dones, infos

    def update_stats(self, observations):
        """
        Update the running mean and standard deviation of observations using EMA.
        """
        # Ensure that observation shape matches self.mean and self.var
        assert observations.shape[1:] == self.mean.shape

        batch_mean = np.mean(observations, axis=0)
        delta = observations - batch_mean
        batch_var = np.mean(delta**2, axis=0)

        self.mean = self.alpha * self.mean + (1.0 - self.alpha) * batch_mean
        self.var = self.alpha * self.var + (1.0 - self.alpha) * batch_var

    def normalize_observations(self, observations):
        """
        Normalize observations using the current EMA mean and standard deviation.
        """
        # Compute the new running standard deviation with a small epsilon for numerical stability
        std = np.sqrt(self.var + self.epsilon)
        return (observations - self.mean) / std


if __name__ == "__main__":
    env = gymz.make("HalfCheetah-v2")
    env = RMVNormalizeVecObservation(env, is_observation_scaling=False)
    print(env.is_observation_scaling)
