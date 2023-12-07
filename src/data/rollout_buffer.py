from typing import Union, NamedTuple, Any

import gymnasium as gymz
import numpy as np
import torch

from pydantic import BaseModel
from pathlib import PurePath
from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer as sb3_RolloutBuffer


class RolloutBufferSamples(BaseModel):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    # Configuration to allow arbitrary types (like torch.Tensor) in Pydantic models
    model_config = {"arbitrary_types_allowed": True}


class RolloutBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent.

    The buffer stores observations, actions, rewards, dones, log probabilities, and value
    estimates for a fixed number of steps across multiple environments. When the buffer exceeds
    its rollout length, it starts overwriting the old data. This buffer supports multiple epochs
    over the data for the Proximal Policy Optimization (PPO) algorithm.

    Attributes:
        rollout_length (int): The length of the rollout. Determines the size of the buffer.
        single_observation_space (spaces.Space): The observation space of the environment.
        single_action_space (spaces.Space): The action space of the environment.
        device (torch.device or str): The device on which to store the tensors (e.g., 'cpu' or
            'cuda').
        n_envs (int): The number of parallel environments.

    Note:
        - The buffer overwrites old data when the rollout_length is exceeded.
        - The buffer can only start yielding data for epochs after it has been filled once,
            indicated by the 'full' flag.
        - Calling the epoch iterator prematurely will include non-overwritten (old) experience
            in the epoch, which may still be useful as PPO aims for policy updates that lead to
            similar policies.
        - An explicit reset is required before starting each new epoch to ensure the buffer is
            aligned with the latest policy.
    """

    def __init__(
        self,
        rollout_length: int,
        single_observation_space: spaces.Space,
        single_action_space: spaces.Space,
        gamma: float,
        gae_lambda: float,
        n_envs: int = 1,
        device: Union[torch.device, str] = "cpu",
    ):
        if not (
            isinstance(single_observation_space, spaces.Box)
            and isinstance(single_action_space, spaces.Box)
        ):
            raise ValueError(
                "Currently only support continuous observation and action space."
            )

        self.rollout_length = rollout_length
        self.single_observation_space = single_observation_space
        self.single_action_space = single_action_space
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_envs = n_envs
        self.device = device

        self.observations, self.actions, self.rewards = None, None, None
        self.dones, self.logprobs, self.values = None, None, None

        self.obs_shape = self.single_observation_space.shape
        self.action_shape = self.single_action_space.shape

        self._is_computation_pending = True
        self.reset()

    def reset(self):
        """
        Resets the buffer.

        This method should be called explicitly before starting each new epoch to align the buffer
        with the latest policy. It clears the buffer and prepares it to store new trajectories.
        """
        self.observations = torch.zeros(
            (self.rollout_length, self.n_envs) + self.obs_shape
        ).to(self.device)
        self.actions = torch.zeros(
            (self.rollout_length, self.n_envs) + self.action_shape
        ).to(self.device)
        self.logprobs = torch.zeros((self.rollout_length, self.n_envs)).to(self.device)
        self.rewards = torch.zeros((self.rollout_length, self.n_envs)).to(self.device)
        self.dones = torch.zeros((self.rollout_length, self.n_envs)).to(self.device)
        self.values = torch.zeros((self.rollout_length, self.n_envs)).to(self.device)

        self.is_full = False
        self.pos = 0
        self.is_computation_pending = True

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        logprob: torch.Tensor,
        value: torch.Tensor,
    ):
        """
        Adds a new experience to the buffer.

        If the buffer is full, it starts overwriting the oldest data.

        Args:
            observation (torch.Tensor): The observed state from the environment.
            action (torch.Tensor): The action taken in response to the observation.
            reward (torch.Tensor): The reward received after taking the action.
            done (torch.Tensor): A flag indicating whether the episode has ended.
            logprob (torch.Tensor): The log probability of the action.
            value (torch.Tensor): The estimated value of the state.
        """
        self.observations[self.pos] = observation.detach().clone()
        self.actions[self.pos] = action.detach().clone()
        self.rewards[self.pos] = reward.detach().clone()
        self.dones[self.pos] = done.detach().clone()
        self.logprobs[self.pos] = logprob.detach().clone()
        self.values[self.pos] = value.detach().clone()

        self.pos += 1

        if self.pos == self.rollout_length:
            self.is_full = True  # Indicates buffer is overflowing
            self.pos = 0

        # Recomputing returns, advantages needed after adding to buffer
        if not self.is_computation_pending:
            self.is_computation_pending = True

    def generate_batches(self, batch_size, final_value, final_done):
        """
        Provides an iterator for generating minibatches of data for training epochs.

        This iterator should be used after the buffer is filled once (indicated by the 'full'
        flag).

        Args:
            batch_size (int): The size of the minibatch.

        Yields:
            RolloutBufferSamples: Minibatches of experience from the buffer.

        Raises:
            ValueError: If the buffer is not full yet.
        """
        if not self.is_full:
            raise ValueError("The buffer is not full yet. Cannot start an epoch.")

        indices = torch.randperm(self.rollout_length * self.n_envs).to(self.device)

        # Avoid recomputing until buffer changes
        if self.is_computation_pending:
            self.compute_advantages_and_returns(final_value, final_done)

        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            # Reshape and index each buffer component to form minibatches
            samples = RolloutBufferSamples(
                observations=self.observations.view(-1, *self.obs_shape)[batch_indices],
                actions=self.actions.view(-1, *self.action_shape)[batch_indices],
                rewards=self.rewards.view(-1)[batch_indices],
                dones=self.dones.view(-1)[batch_indices],
                logprobs=self.logprobs.view(-1)[batch_indices],
                values=self.values.view(-1)[batch_indices],
                advantages=self.advantages.view(-1)[batch_indices],
                returns=self.returns.view(-1)[batch_indices],
            )

            yield samples

    def compute_advantages_and_returns(
        self, final_value: torch.Tensor, final_done: torch.Tensor
    ):
        """
        Computes the Generalized Advantage Estimation (GAE) and the returns for each step in the
        buffer.

        Args:
            final_value (torch.Tensor): The value estimate for the final state, used for
                bootstrapping.
            final_done (torch.Tensor): A flag indicating whether the final state is terminal.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the advantages and returns
                tensors.
        """
        if not self.is_full:
            raise ValueError(
                "The buffer is not full yet. Can NOT compute advantage and returns yet."
            )

        with torch.no_grad():
            final_value = final_value.detach().reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)

            # Initialize the variable for storing last advantage calculation
            last_estimated_advantage = 0

            # Traverse the buffer in reverse order starting from the last inserted position
            for time_step in RolloutBuffer.reverse_traversal_order(
                self.pos, self.rollout_length
            ):
                # Determine if the current step is the last in the sequence
                if time_step == (self.pos - 1) % self.rollout_length:
                    # Use 'is_terminal_state' and 'bootstrap_value' if it's the last step
                    terminal_state_multiplier = 1.0 - final_done.float()
                    next_step_value_estimate = final_value
                else:
                    # Otherwise, use the next step's 'done' flag and value estimate
                    terminal_state_multiplier = 1.0 - self.dones[time_step + 1].float()
                    next_step_value_estimate = self.values[time_step + 1]

                # Calculate the temporal difference error (delta)
                temporal_difference_error = (
                    self.rewards[time_step]
                    + self.gamma * next_step_value_estimate * terminal_state_multiplier
                    - self.values[time_step]
                )

                # Compute the GAE for the current step and store it in 'advantage_estimates'
                advantages[time_step] = last_estimated_advantage = (
                    temporal_difference_error
                    + self.gamma
                    * self.gae_lambda
                    * terminal_state_multiplier
                    * last_estimated_advantage
                )

            # Compute returns as the sum of advantages and value estimates
            returns = advantages + self.values

        self.returns = returns
        self.advantages = advantages

        # Flag computations done for current buffer state
        self.is_computation_pending = False

    def get_explained_variance(self):
        if not self.is_full:
            raise ValueError(
                "Can NOT compute explained variance unless buffer is full."
            )

        if self.is_computation_pending:
            raise ValueError(
                "Buffer state changed and is pending computation of returns and advantages."
            )

        y_pred, y_true = self.values.detach(), self.returns.detach()

        var_y = torch.var(y_true)
        explained_var = (
            torch.tensor(float("nan"))
            if var_y == 0
            else 1 - torch.var(y_true - y_pred) / var_y
        )

        return explained_var.item()

    @staticmethod
    def reverse_traversal_order(pos: int, rollout_length: int):
        """
        Generates a reverse order traversal index list for a rolling buffer.

        The traversal starts from the last inserted position in the buffer and goes backwards
        in time, considering the circular nature of the buffer.

        Args:
            pos (int): The current position in the buffer (next insertion point).
            rollout_length (int): The total length of the buffer.

        Returns:
            List[int]: A list of indices in the order they should be traversed.
        """

        # Calculate the starting index for traversal (last inserted position)
        start_index = (pos - 1) % rollout_length

        # Create a list of indices starting from 'start_index', going backwards,
        # and wrapping around to the end of the buffer if necessary.
        return (
            [start_index]
            + list(range(start_index - 1, -1, -1))
            + list(range(rollout_length - 1, start_index, -1))
        )

    @property
    def is_computation_pending(self):
        return self._is_computation_pending

    @is_computation_pending.setter
    def is_computation_pending(self, value: bool):
        self._is_computation_pending = value
        if value:
            self.advantages = None
            self.returns = None
