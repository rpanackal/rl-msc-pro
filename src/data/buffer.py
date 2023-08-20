from typing import Union, NamedTuple, Optional

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import BaseBuffer, ReplayBufferSamples
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class EpisodicBufferSamples(NamedTuple):
    observations: th.Tensor | list
    actions: th.Tensor | list
    rewards: th.Tensor | list
    dones: th.Tensor | list


class EpisodicBuffer(BaseBuffer):
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
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
    ):
        super(EpisodicBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        self.observations, self.actions, self.rewards = None, None, None
        self.dones = None
        self.generator_ready = False
        self.reset()

        # TODO: Add dones and infos
        # TODO: Use dones instead of episode_starts

    def reset(self) -> None:
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super(EpisodicBuffer, self).reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        if self.full:
            # Shift all data to make room for the new one at the last index
            self.observations[:-1] = self.observations[1:]
            self.actions[:-1] = self.actions[1:]
            self.rewards[:-1] = self.rewards[1:]
            self.dones[:-1] = self.dones[1:]
            # The last index is now free for the new data
            self.pos = self.buffer_size - 1

        # TODO: If termination needs to be handled separately, do that.
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if not self.full and self.pos == self.buffer_size:
            self.full = True

    def sample(
        self,
        batch_size: int,
        desired_length: int | None = None,
        env: VecNormalize | None = None,
    ) -> RolloutBufferSamples:
        # Step 1: Compute episode lengths and starting indices.
        dones_indices = np.flatnonzero(self.swap_and_flatten(self.dones))
        # We account for episode currently being added and starting episode in buffer
        if dones_indices[0] != 0:
            dones_indices = np.hstack((np.array([-1]), dones_indices))
        if dones_indices[-1] != self.pos - 1:
            dones_indices = np.hstack((dones_indices, np.array([self.pos - 1])))
        
        # start_indices are inclusive, while end_indices are exclusive
        start_indices = dones_indices[0:-1] + 1
        end_indices = dones_indices[1:] + 1

        del dones_indices
        
        # Step 2: Filter episodes based on desired length.
        if desired_length:
            episode_lengths = end_indices - start_indices
            valid_indices_idx = np.flatnonzero(episode_lengths >= desired_length)
            if self.pos < batch_size * desired_length or len(valid_indices_idx) == 0:
                raise ValueError(
                    "Not enough time steps filled in buffer to sample batch of the desired length!"
                )
        else:
            valid_indices_idx = list(range(len(start_indices)))
            if len(valid_indices_idx) < batch_size:
                raise ValueError(
                    "Not enough episodes of batch size in the buffer!"
                )

        # Step 3: Randomize sample episodes.
        valid_indices_idx = np.random.permutation(valid_indices_idx)
        sampled_starts = start_indices[valid_indices_idx]
        sampled_ends = end_indices[valid_indices_idx]

        # Step 4: Sample sequences from the buffer.
        (
            sampled_obs,
            sampled_actions,
            sampled_rewards,
            sampled_dones,
        ) = self._get_samples(starts=sampled_starts,
                              ends=sampled_ends, 
                              batch_size=batch_size, 
                              desired_length=desired_length, 
                              env=env)

        # Note: At this point, all sequences are of length `desired_length` if given,
        # then convert them to numpy arrays (since they all have the same length now).
        # Otherwise list of variable length episodes are given.
        if desired_length:
            sampled_obs = self.to_torch(np.array(sampled_obs))
            sampled_actions = self.to_torch(np.array(sampled_actions))
            sampled_rewards = self.to_torch(np.array(sampled_rewards))
            sampled_dones = self.to_torch(np.array(sampled_dones))

        return EpisodicBufferSamples(
            sampled_obs, sampled_actions, sampled_rewards, sampled_dones
        )

    def _get_samples(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        batch_size: int,
        desired_length: int | None = None,
        env: VecNormalize | None = None,
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param starts:
        :param ends:
        :param batch_size:
        :param desired_length:
        :param env: (Unused)
        :return:
        """
        # Step 4: Get flattened view of episodes.
        observations = self.swap_and_flatten(self.observations)
        actions = self.swap_and_flatten(self.actions)
        rewards = self.swap_and_flatten(self.rewards)
        dones = self.swap_and_flatten(self.dones)

        # Allow start sampling anywhere in the episode to get desired_length trajectory if given.
        sampled_obs, sampled_actions, sampled_rewards, sampled_dones = [], [], [], []
        
        # Initialize a counter for starts and ends
        counter = 0
        
        # Loop until you have enough samples
        while counter < batch_size:
            # Use modulo indexing to loop through starts and ends
            start = starts[counter % len(starts)]
            end = ends[counter % len(ends)]
            
            effective_start = (
                np.random.randint(start, end - desired_length + 1)
                if desired_length
                else start
            )
            
            # TODO: Add any normalization applied in environment if necessary
            effective_end = effective_start + desired_length if desired_length else end
            
            sampled_obs.append(observations[effective_start:effective_end].copy())
            sampled_actions.append(actions[effective_start:effective_end].copy())
            sampled_rewards.append(rewards[effective_start:effective_end].copy())
            sampled_dones.append(dones[effective_start:effective_end].copy())
            
            # Increment the counter
            counter += 1

        return sampled_obs, sampled_actions, sampled_rewards, sampled_dones
