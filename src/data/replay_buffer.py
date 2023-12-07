from typing import Union, NamedTuple, Any

import gymnasium as gymz
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer, RolloutBuffer
from ..envs.wrappers.normalization import RMVNormalizeVecObservation
from src.utils import get_observation_dim
from pydantic import BaseModel
from pathlib import PurePath


class EpisodicReplayBufferSamples(BaseModel):
    observations: Union[th.Tensor, list[np.ndarray]]
    next_observations: Union[th.Tensor, list[np.ndarray], None]
    actions: Union[th.Tensor, list[np.ndarray]]
    rewards: Union[th.Tensor, list[np.ndarray]]
    dones: Union[th.Tensor, list[np.ndarray]]
    contexts: Union[th.Tensor, list[np.ndarray], None]

    model_config = {"arbitrary_types_allowed": True}


class EpisodicReplayBuffer(BaseBuffer):
    """
    EpisodicBuffer is used to append experience at each time step
    and sample random batch of episodes from it.
    The``buffer_size`` coppesponds to transitions collected in total.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: If True, next observations given are ignored.
    :param shift_on_overflow: If true, during overflow the buffer is shifted left by 1 step, else
        we overwrite existing entries starting from index 0 of buffer.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        include_context: bool = False,
    ):
        super(EpisodicReplayBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        self.observations, self.actions, self.rewards = None, None, None
        self.next_observations, self.dones, self.contexts = None, None, None

        self.optimize_memory_usage = optimize_memory_usage
        self.include_context = include_context

        # To handle carl environment observation space
        if (
            isinstance(self.observation_space, spaces.Dict)
            and "obs" in self.observation_space.keys()
        ):
            self.obs_shape = self.observation_space["obs"].shape

        if self.include_context:
            self.context_dim = np.prod(self.observation_space["context"].shape)
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if not self.optimize_memory_usage:
            self.next_observations = np.zeros(
                (self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32
            )

        if self.include_context:
            self.contexts = np.zeros(
                (self.buffer_size, self.n_envs, self.context_dim), dtype=np.float32
            )

        super(EpisodicReplayBuffer, self).reset()

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]] = None,
        context: Union[np.ndarray, None] = None,
    ) -> None:
        """
        Adds a new experience tuple to the episodic buffer.

        Args:
            obs: The observation at the current timestep.
            next_obs: The observation at the next timestep.
            action: The action taken at the current timestep.
            reward: The reward received at the current timestep.
            done: A boolean flag indicating if the episode terminated at this timestep.
            infos: Additional info, currently unused.
        """

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        # TODO: If termination needs to be handled separately, do that.
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if not self.optimize_memory_usage:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        if self.include_context:
            if context is None:
                raise ValueError("Context missing for trasition.")
            self.contexts[self.pos] = np.array(context).copy()

        self.pos += 1

        if self.pos == self.buffer_size:
            self.full = True  # Indicates buffer is overflowing
            self.pos = 0

    def sample(
        self,
        batch_size: int,
        desired_length: Union[int, None] = None,
        env: Union[RMVNormalizeVecObservation, gymz.Env, None] = None,
    ) -> EpisodicReplayBufferSamples:
        """
        Samples a random batch of episodes from the episodic buffer.

        Args:
            batch_size: Number of episodes to sample.
            desired_length: Optional desired length for each sampled episode.
            env: Optional environment for observation normalization.

        Returns:
            (EpisodicBufferSamples): An EpisodicBufferSamples object containing the sampled episodes.
        """
        # Step 1: Compute episode lengths and starting indices.
        dones = self.dones.copy() if self.full else self.dones[: self.pos].copy()

        # Make sure while flattening dones, we will not join episodes from
        # different environments in buffer and treat episodes from
        # different cycles differently.
        #   - Assume last transition of progressing episode is done
        dones[self.pos - 1, :] = 1
        #   - Assume last transition in buffer is done, if full.
        if self.full:
            dones[-1, :] = 1

        dones_indices = np.flatnonzero(self.swap_and_flatten(dones))
        # We account for starting episode in flattened buffer
        if dones_indices[0] != 0:
            dones_indices = np.hstack((np.array([-1]), dones_indices))

        # start_indices are inclusive, while end_indices are exclusive
        start_indices = dones_indices[0:-1] + 1
        end_indices = dones_indices[1:] + 1

        del dones_indices

        # Step 2: Filter episodes based on desired length.
        if desired_length:
            episode_lengths = end_indices - start_indices
            valid_indices_idx = np.flatnonzero(episode_lengths >= desired_length)
            if not self.full and self.pos < batch_size * desired_length:
                raise ValueError(
                    "Not enough episodes of batch size in the buffer! Try, start learning"
                    + "after a larger number of timesteps."
                )
        else:
            valid_indices_idx = list(range(len(start_indices)))
            if len(valid_indices_idx) < batch_size:
                raise ValueError(
                    "Not enough episodes of batch size in the buffer! Optionally, start learning"
                    + "after a larger number of timesteps."
                )

        # Step 3: Randomize sample episodes.
        valid_indices_idx = np.random.permutation(valid_indices_idx)
        sampled_starts = start_indices[valid_indices_idx]
        sampled_ends = end_indices[valid_indices_idx]

        # Step 4: Sample sequences from the buffer.
        return self._get_samples(
            starts=sampled_starts,
            ends=sampled_ends,
            batch_size=batch_size,
            desired_length=desired_length,
            env=env,
        )

    def _get_samples(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        batch_size: int,
        desired_length: Union[int, None] = None,
        env: Union[RMVNormalizeVecObservation, None] = None,
    ) -> EpisodicReplayBufferSamples:
        """
        Internal method to retrieve samples based on given episode start and end indices.

        Args:
            starts: Array of starting indices for the episodes to sample.
            ends: Array of ending indices for the episodes to sample.
            batch_size: Number of episodes to sample.
            desired_length: Optional desired length for each sampled episode.
            env: Optional environment for observation normalization.

        Returns:
            (EpisodicBufferSamples): An EpisodicBufferSamples object containing the sampled episodes.
        """
        # Step 4: Get flattened view of episodes.
        observations = self.observations if self.full else self.observations[: self.pos]
        actions = self.actions if self.full else self.actions[: self.pos]
        rewards = self.rewards if self.full else self.rewards[: self.pos]
        dones = self.dones if self.full else self.dones[: self.pos]

        observations = self.swap_and_flatten(observations)
        actions = self.swap_and_flatten(actions)
        rewards = self.swap_and_flatten(rewards)
        dones = self.swap_and_flatten(dones)

        # Allow start sampling anywhere in the episode to get desired_length trajectory if given.
        sampled_obs, sampled_actions, sampled_rewards, sampled_dones = [], [], [], []

        if not self.optimize_memory_usage:
            next_observations = (
                self.next_observations
                if self.full
                else self.next_observations[: self.pos]
            )
            next_observations = self.swap_and_flatten(next_observations)
            sampled_next_obs = []

        if self.include_context:
            contexts = self.contexts if self.full else self.contexts[: self.pos]
            contexts = self.swap_and_flatten(contexts)
            sampled_contexts = []

        # Initialize a counter for starts and ends
        counter = 0

        # Loop until you have enough samples
        while counter < batch_size:
            # Use modulo indexing to loop through starts and ends
            start = starts[counter % len(starts)]
            end = ends[counter % len(ends)]

            # Start from anywhere in the episode that can give desired length trajectories
            # if desired_length given.
            effective_start = (
                np.random.randint(start, end - desired_length + 1)
                if desired_length
                else start
            )

            # TODO: Add any normalization applied in environment if necessary
            effective_end = effective_start + desired_length if desired_length else end

            sampled_obs.append(
                self.normalize_obs(
                    observations[effective_start:effective_end].copy(), env=env
                )
            )
            sampled_actions.append(actions[effective_start:effective_end].copy())
            sampled_rewards.append(rewards[effective_start:effective_end].copy())
            sampled_dones.append(dones[effective_start:effective_end].copy())

            if not self.optimize_memory_usage:
                sampled_next_obs.append(
                    self.normalize_obs(
                        next_observations[effective_start:effective_end].copy(), env=env
                    )
                )

            if self.include_context:
                sampled_contexts.append(contexts[effective_start:effective_end].copy())

            # Increment the counter
            counter += 1

        # Note: At this point, all sequences are of length `desired_length` if given,
        # then convert them to numpy arrays (since they all have the same length now).
        # Otherwise list of variable length episodes are given.
        if desired_length:
            sampled_obs = self.to_torch(np.array(sampled_obs))
            sampled_actions = self.to_torch(np.array(sampled_actions))
            sampled_rewards = self.to_torch(np.array(sampled_rewards))
            sampled_dones = self.to_torch(np.array(sampled_dones))

            if not self.optimize_memory_usage:
                sampled_next_obs = self.to_torch(np.array(sampled_next_obs))

            if self.include_context:
                sampled_contexts = self.to_torch(np.array(sampled_contexts))

        return EpisodicReplayBufferSamples(
            observations=sampled_obs,
            next_observations=sampled_next_obs
            if not self.optimize_memory_usage
            else None,
            actions=sampled_actions,
            rewards=sampled_rewards,
            dones=sampled_dones,
            contexts=sampled_contexts if self.include_context else None,
        )

    def get_last_episode(
        self,
        env: Union[RMVNormalizeVecObservation, gymz.Env, None] = None,
    ):
        """
        Retrieves the last episode for each environment in the buffer.

        Args:
            env: Optional environment for observation normalization.

        Returns:
            An EpisodicBufferSamples object containing the last episodes.
        """
        observations = []
        actions = []
        rewards = []
        dones = []

        if not self.optimize_memory_usage:
            next_observations = []

        if self.include_context:
            contexts = []

        for env_no in range(self.n_envs):
            # Compute start of progressing episode
            dones_indices = np.flatnonzero(self.dones[: self.pos, env_no])
            # We account for starting episode in flattened buffer
            if len(dones_indices) == 0 or dones_indices[0] != 0:
                dones_indices = np.hstack((np.array([-1]), dones_indices))

            start_last_ep = dones_indices[-1] + 1 if len(dones_indices) else 0

            # TODO: What happens if the last entry in the buffer was a done transitions ?
            observations.append(
                self.normalize_obs(
                    self.observations[start_last_ep : self.pos, env_no], env=env
                )
            )
            actions.append(self.actions[start_last_ep : self.pos, env_no])
            rewards.append(self.rewards[start_last_ep : self.pos, env_no])
            dones.append(self.dones[start_last_ep : self.pos, env_no])

            if not self.optimize_memory_usage:
                next_observations.append(
                    self.next_observations[start_last_ep : self.pos, env_no]
                )

            if self.include_context:
                contexts.append(self.contexts[start_last_ep : self.pos, env_no])

        return EpisodicReplayBufferSamples(
            observations=observations,
            next_observations=next_observations
            if not self.optimize_memory_usage
            else None,
            actions=actions,
            rewards=rewards,
            dones=dones,
            contexts=contexts if self.include_context else None,
        )

    def normalize_obs(
        self,
        obs: np.ndarray,
        env: Union[RMVNormalizeVecObservation, gymz.Env, None] = None,
    ):
        """The environment that is wrapped by RMVNormalizeVecObservation and
        is not scaling observations while sampling from environment will be
        then normalized during sampling from buffer.

        Args:
            obs (_type_): _description_
                shape: (seq_len, obs_dim)
            env (Union[RMVNormalizeVecObservation, gymz.Env, None]): If None, no scaling is done.

        Returns:
            np.ndarray: Normalized observations.
        """
        # TODO: Observations here is for a sequence of time steps.
        if isinstance(env, RMVNormalizeVecObservation) and getattr(
            env, "is_observation_scaling", False
        ):
            return env.normalize_observations(obs)
        return obs

    def save(self, path: PurePath):
        """Save buffer to disk in .npz format

        Args:
            path (PurePath): Path to save buffer at.
        """
        path = path if isinstance(path, PurePath) else PurePath(path)

        observations = self.observations if self.full else self.observations[: self.pos]
        actions = self.actions if self.full else self.actions[: self.pos]
        rewards = self.rewards if self.full else self.rewards[: self.pos]
        dones = self.dones if self.full else self.dones[: self.pos]

        # Some measures to make sure flattening does mix episodes across environments
        # Assume last filled position is also the end
        dones[self.pos - 1, :] = 1
        # Assume last transition in buffer is done, if full.
        if self.full:
            dones[-1, :] = 1

        observations = self.swap_and_flatten(observations)
        actions = self.swap_and_flatten(actions)
        rewards = self.swap_and_flatten(rewards)
        dones = self.swap_and_flatten(dones)

        buffer = {
            "observations": observations,
            "rewards": rewards,
            "actions": actions,
            "dones": dones,
        }

        if not self.optimize_memory_usage:
            next_observations = self.next_observations if self.full else self.next_observations[: self.pos]
            buffer["next_observations"] = self.swap_and_flatten(next_observations)

        if self.include_context:
            contexts = self.contexts if self.full else self.contexts[: self.pos]
            buffer["contexts"] = self.swap_and_flatten(contexts)

        np.savez((path / "buffer.npz"), **buffer)

    def load(self, path: PurePath):
        preloaded_buffer = np.load(path)

        # ! Currently loading from the prev save method that didnt swap and flatten buffer items
        observations = preloaded_buffer["observations"]
        rewards = preloaded_buffer["rewards"]
        actions = preloaded_buffer["actions"]
        dones = preloaded_buffer["dones"]

        self.observations = observations
        self.rewards = rewards
        self.actions = actions
        self.dones = dones

        if not self.optimize_memory_usage:
            self.next_observations = preloaded_buffer["next_observations"]

        if self.include_context:
            self.contexts = preloaded_buffer["contexts"]

        # Enable sampling from the who buffer after loading
        self.full = True

    @staticmethod
    def reverse_swap_and_flatten(arr: np.ndarray, n_envs=1) -> np.ndarray:
        """
        Reverse the operation of swap_and_flatten.
        Reshape and then swap axes 1 (n_envs) and 0 (buffer_size)
        to convert shape from [n_steps * n_envs, ...] back to [n_steps, n_envs, ...]

        :param arr: Flattened array
        :param original_shape: The original shape of the array before swap_and_flatten
        :return: Reshaped array
        """
        reshaped_arr = arr.reshape(n_envs, -1, *arr.shape[1:])
        return reshaped_arr.swapaxes(0, 1)
