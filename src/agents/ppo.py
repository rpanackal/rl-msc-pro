import time
from typing import Union
import math
import gymnasium as gymz
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from ..assets.layers.feedforward import OrthoConstantLinear
from ..utils import get_action_dim, get_observation_dim, is_vector_env
from .core import GenericAgent
from pathlib import PurePath
from pydantic import BaseModel
from ..data.rollout_buffer import RolloutBufferSamples, RolloutBuffer


class PPOActor(nn.Module):
    def __init__(self, envs, observation_dim, expanse_dim=64):
        super().__init__()
        # Initialize the actor's networks: mean and logstd
        action_dim = get_action_dim(envs)

        self.actor_mean = self._build_network(observation_dim, action_dim, expanse_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def _build_network(self, input_dim, output_dim, expanse_dim):
        """Builds the neural network for the PPO actor.

        This network computes the mean of the action distribution.
        The network architecture is defined here with 'expanse_dim' hidden layers.

        Args:
            input_dim (int): Dimensionality of the input feature space.
            output_dim (int): Dimensionality of the output action space.
            expanse_dim (int): Size of the hidden layers.

        Returns:
            nn.Sequential: The constructed neural network.
        """
        return nn.Sequential(
            OrthoConstantLinear(input_dim, expanse_dim),
            nn.Tanh(),
            OrthoConstantLinear(expanse_dim, expanse_dim),
            nn.Tanh(),
            OrthoConstantLinear(expanse_dim, output_dim, std=0.01),
        )

    def forward(self, x):
        """Forward pass through the network to compute action mean and
        log standard deviation..

        Args:
            x (Tensor): Input state or feature tensor.

        Returns:
            tuple[Tensor, Tensor]: The computed mean of the action distribution and log std.
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def get_action(self, x, action=None):
        """
        Generates an action given the current state or evaluates a provided action.

        This method serves two purposes:
        1. If no action is provided (action=None), it samples a new action based on
            the current policy. This is typically used during training when the agent needs to explore the action space.
        2. If an action is provided, it calculates the log probability and entropy for that
            specific action.This can be useful for evaluating the likelihood of certain actions
            under the current policy, often used during policy updates or for diagnostic purposes.

        Args:
            x (Tensor): The input state or observation for which the action needs to be generated.
                It should be a tensor of appropriate shape that matches the input dimensionality of
                the actor network.
            action (Tensor, optional): An optional action tensor. If provided, the method will
                calculate its log probability and entropy under the current policy. If None, a new action will be sampled. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing:
                - action (Tensor): The sampled action (if no action was provided) or the provided
                    action.
                - log_prob (Tensor): The log probability of the action. It quantifies how likely
                    the action is under the current policy.
                - entropy (Tensor): The entropy of the action distribution at the input state. High
                    entropy indicates more randomness in the action selection, which can be a
                    sign of exploration or uncertainty in the policy.
        """
        action_mean, action_logstd = self(x)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        # Sample a new action if none is provided
        if action is None:
            action = probs.sample()

        # Calculate log probability of the action
        log_prob = probs.log_prob(action).sum(1)

        # Compute entropy of the action distribution
        entropy = probs.entropy().sum(1)

        return action, log_prob, entropy


class PPOCritic(nn.Module):
    def __init__(self, observation_dim, expanse_dim=64):
        """
        Initialize the PPO Critic network.

        The PPO critic estimates the value function V(s) for a given state. This helps
        in policy updates by providing a baseline to calculate the advantage function.

        Args:
            observation_dim (int): The dimension of the feature (observation) space.
            expanse_dim (int): The dimension of the hidden layers in the network.
        """
        super().__init__()
        self.observation_dim = observation_dim
        self.expanse_dim = expanse_dim

        # Define the neural network layers
        self.fc1 = OrthoConstantLinear(observation_dim, expanse_dim)
        self.fc2 = OrthoConstantLinear(expanse_dim, expanse_dim)
        self.fc3 = OrthoConstantLinear(expanse_dim, 1)

    def forward(self, x):
        """
        Forward pass through the network to compute the state value.

        Args:
            x (Tensor): The input state or observation tensor.

        Returns:
            Tensor: The estimated value of the state.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def model_twin(self):
        """
        Create and return a twin of the current model.

        This method is useful for creating a duplicate of the current critic model
        with the same initial parameters and configuration. This can be beneficial
        in scenarios where multiple critic models are required, such as for target
        network updates in certain reinforcement learning algorithms.

        Returns:
            PPOCritic: A new instance of PPOCritic with the same configuration.
        """
        return self.__class__(self.observation_dim, self.expanse_dim)


class PPOAgent(GenericAgent):
    """
    A Proximal Policy Optimization (PPO) agent for reinforcement learning in continuous action spaces.

    This class implements the PPO algorithm, which is a policy gradient method for training
    deep reinforcement learning agents. It is specifically designed to work with environments
    that have continuous action spaces. The agent contains separate neural networks for the
    policy (actor) and value function (critic). The implementation includes key features of
    PPO such as clipping in the objective function, actor and critic network updates, and
    support for vectorized environments for efficient training.

    The agent assumes that the action space is continuous. It uses the Adam optimizer
    for updating the parameters of the actor and critic networks.

    Note:
        The implementation is tailored to work with vectorized environments, which allow
        for batch processing of environment steps. This makes training more efficient.
    """

    def __init__(
        self,
        envs: gymz.vector.VectorEnv,
        learning_rate: float,
        device: torch.device,
        writer: Union[SummaryWriter, None] = None,
        log_freq: int = 100,
        expanse_dim: int = 256,
        seed: int = 1,
    ):
        """
        Initialize the PPOAgent instance.

        Sets up the actor and critic networks, optimizer, and other necessary parameters for
        training the PPO agent in a vectorized environment setting.

        Args:
            envs (gymz.vector.VectorEnv): Vectorized environments for training the agent.
            learning_rate (float): Learning rate for the optimizer.
            device (torch.device): Device (CPU or GPU) to perform computations.
            writer (Union[SummaryWriter, None]): Optional TensorBoard writer for logging.
            log_freq (int): Frequency of logging metrics.
            expanse_dim (int): Dimension of the hidden layers in the actor and critic networks.
            seed (int): Random seed for reproducibility.
        """
        super().__init__(envs)
        self.device = device
        self.writer = writer
        self.log_freq = log_freq
        self.seed = seed

        # Ensure the environment is vectorized for efficient batch processing
        self.is_vector_env: bool = is_vector_env(envs)
        assert self.is_vector_env, "Environment not vectorized"

        # Verify that the action space is continuous
        self.single_observation_space = self.envs.single_observation_space
        self.single_action_space = self.envs.single_action_space
        self.n_envs = self.envs.num_envs

        assert isinstance(self.single_action_space, gymz.spaces.Box), ValueError(
            f"only continuous action space is supported, given {self.single_action_space}"
        )

        # Determine dimensions of observation and action spaces
        self.observation_dim = get_observation_dim(self.envs)
        self.action_dim = get_action_dim(self.envs)

        # Initialize actor and critic networks
        self.actor = PPOActor(self.envs, self.observation_dim, expanse_dim).to(device)
        self.critic = PPOCritic(self.observation_dim, expanse_dim).to(device)

        # Set up the optimizer for the actor and critic networks
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate,
        )

    def sample_action(self, state, action=None):
        """Sample an action from the policy (actor), and get value estimate from critic."""
        action, log_prob, entropy = self.actor.get_action(state, action)
        return action, log_prob, entropy

    def initialize(
        self,
        total_timesteps: int,
        rollout_length: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        clip_coef: float,
        clip_vloss: bool,
        norm_adv: bool,
        target_kl: float,
        batch_size: int,
        n_epochs: int,
        anneal_lr: bool = False,
    ):
        """
        Initialize or reset the training configuration and hyperparameters.

        This method sets up the necessary parameters and objects needed before starting or
        restarting the training process.

        Args:
            total_timesteps (int): Total number of timesteps for which the agent will be trained.
            rollout_length (int): Number of steps to collect per environment before updating the
                policy.
            gamma (float): Discount factor for future rewards.
            gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
            ent_coef (float): Coefficient for the entropy term in the loss calculation.
            vf_coef (float): Coefficient for the value function loss.
            max_grad_norm (float): Maximum norm for gradient clipping.
            clip_coef (float): Coefficient for PPO's policy clipping.
            clip_vloss (bool): Flag to determine if value loss should be clipped.
            norm_adv (bool): Flag to determine if advantages should be normalized.
            target_kl (float): Target KL divergence for early stopping.
            batch_size (int): Batch size for training updates.
            n_epochs (int): Number of epochs to train on each batch of data.
            anneal_lr (bool): Flag to determine if learning rate should be annealed over training.

        The method performs the following steps:
        - Sets up the training hyperparameters.
        - Initializes the rollout buffer for storing experiences.
        - Configures the learning rate scheduler if annealing is enabled.
        """
        # Set training hyperparameters
        self.total_timesteps = total_timesteps

        self.rollout_length = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.norm_adv = norm_adv
        self.target_kl = target_kl
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.anneal_lr = anneal_lr

        # Calculate the buffer size and total number of updates
        self.buffer_size = int(self.n_envs * self.rollout_length)
        n_batches = math.ceil(self.buffer_size / self.batch_size)
        total_data_cycles = int(self.total_timesteps // self.rollout_length)
        self.total_n_updates = int(total_data_cycles * self.n_epochs * n_batches)

        # Initialize the rollout buffer for storing experiences
        self.buffer = RolloutBuffer(
            rollout_length=self.rollout_length,
            single_observation_space=self.single_observation_space,
            single_action_space=self.single_action_space,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            device=self.device,
        )

        # Set up the learning rate scheduler if annealing is enabled
        if self.anneal_lr:
            self.lr_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1,
                end_factor=0,
                total_iters=self.total_n_updates,
            )

        # Initialize the global step counter
        self.global_step = None

    def preprocess_experience(
        self,
        curr_obs,
        curr_done,
        action,
        next_obs,
        reward,
        termination,
        truncation,
        logprob,
        value,
    ):
        """
        Preprocesses the raw experience data collected from the environment.

        Args:
            curr_obs: Current observations from the environment.
            curr_done: Flags indicating if the current state is terminal.
            action: Actions taken in the current state.
            next_obs: Observations for the next state.
            reward: Rewards received after taking the action.
            termination: Flags indicating if the episode has ended.
            truncation: Flags indicating if the episode was truncated.
            logprob: Log probabilities of the actions taken.
            value: Estimated values of the current states.

        Returns:
            dict: A dictionary containing processed experience data with keys representing
                the data types (observations, actions, rewards, dones, etc.).
        """
        next_done = np.logical_or(termination, truncation)

        return {
            "curr_obs": curr_obs,
            "curr_done": curr_done,
            "action": action,
            "next_obs": torch.as_tensor(next_obs, device=self.device),
            "reward": torch.as_tensor(reward, device=self.device).view(-1),
            "next_done": torch.as_tensor(next_done, device=self.device),
            "logprob": logprob,
            "value": value.flatten(),
        }

    def update_agent(self, experience):
        """
        Update the agent based on the collected experience.

        This method handles the training update of the agent's policy (actor) and value function
        (critic) using the Proximal Policy Optimization (PPO) algorithm. It involves multiple
        epochs of training over minibatches of experience data.

        Args:
            experience (tuple): A tuple containing the current observations, done flags, actions,
            next observations, rewards, next done flags, log probabilities of the actions, and value estimates.

        The method performs the following steps:
        - Adds the collected experience to the buffer.
        - Checks if the buffer is full to proceed with training.
        - Iterates over the training epochs and minibatches.
        - Computes new action log probabilities and entropy.
        - Calculates the policy gradient loss, value loss, and entropy loss.
        - Performs backpropagation and updates the neural network parameters.
        - Implements early stopping based on KL divergence.
        - Logs various training metrics for performance monitoring.
        """
        # Add experience into rollout buffer
        self.buffer.add(
            observation=experience["curr_obs"],
            action=experience["action"],
            reward=experience["reward"],
            done=experience["curr_done"],
            logprob=experience["logprob"],
            value=experience["value"],
        )

        # Return if rollout in progress
        if not self.buffer.is_full:
            return

        clip_fracs = []
        # Get batches per epoch for update.
        for _ in range(self.n_epochs):
            for batch in self.buffer.generate_batches(
                batch_size=self.batch_size,
                final_value=self.critic(experience["next_obs"]),
                final_done=experience["next_done"],
            ):
                batch: RolloutBufferSamples

                # Sample new actions based on current policy and calculate their log probabilities
                # and entropy
                _, new_logprob, entropy = self.sample_action(
                    batch.observations, batch.actions
                )

                # Compute the ratio of new to old policy probabilities
                log_ratio = new_logprob - batch.logprobs
                ratio = log_ratio.exp()

                # Calculate the approximate KL divergence and clip fraction without gradient
                # tracking
                with torch.no_grad():
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = self.compute_approx_kl_divergence(
                        ratio=ratio, log_ratio=log_ratio
                    )
                    clip_fracs += [self.compute_clip_fraction(ratio)]

                # Normalize advantages if specified
                if self.norm_adv:
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )

                # Compute the total loss from policy gradient loss, entropy loss, and value loss
                pg_loss = self.compute_actor_loss(
                    advantages=batch.advantages, ratio=ratio
                )
                entropy_loss = self.compute_entropy_loss(entropy)
                v_loss = self.compute_critic_loss(batch)

                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                # Perform backpropagation and update the actor and critic networks
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                if self.anneal_lr:
                    self.lr_scheduler.step()

            # Check for early stopping based on KL divergence
            do_early_stopping = (
                self.target_kl is not None and approx_kl > self.target_kl
            )
            if do_early_stopping:
                break

        # Calculate the explained variance to evaluate the critic's performance
        explained_var = self.buffer.get_explained_variance()

        # Reset the buffer to collect new experiences for the next rollout
        self.buffer.reset()

        # Log training metrics at final update for experience collected from a rollout
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
        self.writer.add_scalar(
            "losses/entropy_loss", entropy_loss.item(), self.global_step
        )
        self.writer.add_scalar(
            "charts/old_approx_kl", old_approx_kl.item(), self.global_step
        )
        self.writer.add_scalar("charts/approx_kl", approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), self.global_step)
        self.writer.add_scalar(
            "charts/explained_variance", explained_var, self.global_step
        )
        self.writer.add_scalar(
            "charts/learning_rate",
            self.optimizer.param_groups[0]["lr"],
            self.global_step,
        )

    def compute_approx_kl_divergence(self, ratio, log_ratio):
        """
        Compute the approximate Kullback-Leibler (KL) divergence.

        This method estimates the KL divergence, a measure of how one probability distribution
        diverges from a second, expected probability distribution. In the context of PPO, it's used
        to measure the difference between the new and old policies.

        Args:
            ratio (Tensor): The ratio of the probabilities under the new policy to the old policy.
            log_ratio (Tensor): The log of the aforementioned ratio.

        Returns:
            float: The mean of the approximate KL divergence across all elements in the batch.

        Reference:
            - http://joschu.net/blog/kl-approx.html
        """
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        approx_kl = ((ratio - 1) - log_ratio).mean()
        return approx_kl

    def compute_clip_fraction(self, ratio):
        """
        Compute the fraction of ratios that are clipped.

        This method calculates the proportion of 'ratios' that fall outside the range specified by
        `self.clip_coef`. This is a part of the PPO clipping mechanism which limits the updates to
        the policy (actor) network, encouraging more gradual policy updates.

        Args:
            ratio (Tensor): The ratio of probabilities under the new policy to the old policy.

        Returns:
            float: The fraction of 'ratio' values that were clipped.
        """
        # Calculate the fraction of ratios that exceed the clipping threshold
        clip_frac = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
        return clip_frac

    def compute_entropy_loss(self, entropy):
        """
        Compute the mean entropy loss.

        This method calculates the mean entropy of the policy's action distribution. Entropy is a
        measure of randomness or unpredictability in the policy's action selection. Higher entropy
        generally encourages exploration by making the policy less deterministic.

        Args:
            entropy (Tensor): The entropy of the policy's action distribution for each element in
            the batch.

        Returns:
            float: The mean entropy across all elements in the batch.
        """
        return entropy.mean()

    def compute_critic_loss(self, batch: RolloutBufferSamples):
        """
        Compute the loss for the critic network.

        This method calculates the mean squared error between the predicted value estimates of
        the critic network and the actual returns. It supports optional value clipping, which
        can help stabilize training by limiting large updates to the critic.

        Args:
            batch (RolloutBufferSamples): A batch of experiences, including observations and
                returns.

        Returns:
            float: The computed critic loss, which is either clipped or unclipped based on the
                configuration.
        """
        new_value = self.critic(batch.observations).view(-1)

        if self.clip_vloss:
            # Unclipped value loss
            v_loss_unclipped = (new_value - batch.returns) ** 2

            # Clipped value loss
            v_clipped = batch.values + torch.clamp(
                new_value - batch.values,
                -self.clip_coef,
                self.clip_coef,
            )
            v_loss_clipped = (v_clipped - batch.returns) ** 2

            # Use the maximum of clipped and unclipped losses
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            # If clipping is not used, calculate the standard mean squared error
            v_loss = 0.5 * ((new_value - batch.returns) ** 2).mean()
        return v_loss

    def compute_actor_loss(self, ratio, advantages):
        """
        Compute the loss for the actor (policy) network.

        This method calculates the policy gradient loss, which is essential for updating the policy
        network in PPO. It employs the PPO clipping technique to limit the updates to the policy,
        preventing too large updates and promoting more stable training.

        Args:
            ratio (Tensor): The ratio of probabilities under the new policy to the old policy.
            advantages (Tensor): The advantage estimates for each action in the batch.

        Returns:
            float: The computed policy gradient loss, considering the clipping mechanism.
        """
        # Calculate the policy gradient loss with clipping
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1 - self.clip_coef, 1 + self.clip_coef
        )

        # Use the maximum of the two losses to enforce clipping
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        return pg_loss

    def train(
        self,
        total_timesteps: int,
        rollout_length: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        clip_coef: float,
        clip_vloss: bool,
        norm_adv: bool,
        target_kl: float,
        batch_size: int,
        n_epochs: int,
        anneal_lr: bool = False,
    ):
        """
        Execute the main training loop for the PPO agent.

        This method orchestrates the training process, involving environment interaction,
        data collection, policy updating, and logging.

        Args:
            total_timesteps (int): The total number of timesteps to train for.
            rollout_length (int): The number of steps to collect per environment before updating
                the policy.
            gamma (float): The discount factor.
            gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
            ent_coef (float): Coefficient for the entropy term in the loss calculation.
            vf_coef (float): Coefficient for the value function loss.
            max_grad_norm (float): Maximum gradient norm for gradient clipping.
            clip_coef (float): PPO clipping coefficient.
            clip_vloss (bool): Whether to clip the value function loss.
            norm_adv (bool): Whether to normalize advantages.
            target_kl (float): Target KL divergence threshold for early stopping.
            batch_size (int): The number of samples per minibatch.
            n_epochs (int): The number of epochs to train on the collected data.
            anneal_lr (bool, optional): Whether to anneal the learning rate.

        The training involves:
        - Initializing the training setup.
        - Iteratively collecting experiences from the environment.
        - Performing PPO updates using the collected experiences.
        - Logging important metrics and episodic information.
        """
        # Initialize training configurations and hyperparameters
        self.initialize(
            total_timesteps=total_timesteps,
            rollout_length=rollout_length,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            clip_coef=clip_coef,
            clip_vloss=clip_vloss,
            norm_adv=norm_adv,
            target_kl=target_kl,
            batch_size=batch_size,
            n_epochs=n_epochs,
            anneal_lr=anneal_lr,
        )

        # Start timer
        start_time = time.time()

        # Reset the environment and initialize states
        curr_obs, _ = self.envs.reset(seed=self.seed)
        curr_obs = torch.as_tensor(curr_obs, device=self.device)
        curr_done = torch.zeros(self.n_envs).to(self.device)

        # Main training loop
        for self.global_step in range(self.total_timesteps):
            # Collect experiences from the environment
            with torch.no_grad():
                action, logprob, _ = self.sample_action(curr_obs)
                value = self.critic(curr_obs)

            next_obs, reward, termination, truncation, infos = self.envs.step(
                action.cpu().numpy()
            )

            # Preprocess and organize the collected experience
            experience = self.preprocess_experience(
                curr_obs=curr_obs,
                curr_done=curr_done,
                action=action,
                next_obs=next_obs,
                reward=reward,
                termination=termination,
                truncation=truncation,
                logprob=logprob,
                value=value,
            )

            # Update the agent using the collected experience
            self.update_agent(experience)

            # Prepare for the next iteration
            curr_obs = experience["next_obs"]
            curr_done = experience["next_done"]

            # TODO: Check if info content is as expected
            # Log episodic information if available
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={self.global_step}, episodic_return={info['episode']['r']}"
                        )
                        self.writer.add_scalar(
                            "train/episodic_return",
                            info["episode"]["r"],
                            self.global_step,
                        )
                        self.writer.add_scalar(
                            "train/episodic_length",
                            info["episode"]["l"],
                            self.global_step,
                        )

            # Log steps per second (SPS) every self.log_freq steps
            if self.global_step % self.log_freq == 0:
                print("SPS:", int(self.global_step / (time.time() - start_time)))
                if self.writer:
                    self.writer.add_scalar(
                        "train/SPS",
                        int(self.global_step / (time.time() - start_time)),
                        self.global_step,
                    )

    def evaluate(
        self, n_episodes: int, test_envs: Union[gymz.vector.VectorEnv, None] = None
    ):
        """
        Evaluate the trained PPO agent over a number of episodes.

        Args:
            n_episodes (int): Number of episodes to run for evaluation.

        Returns:
            dict: A dictionary with environment index as keys and average returns as values.
        """
        # By default self.env will be used if test_envs not passed
        test_envs = test_envs or self.envs
        n_envs = test_envs.num_envs

        # Initialize tensors to store episode returns for each envs
        episode_returns = torch.zeros((n_envs, n_episodes), device=self.device)

        for ep_idx in range(n_episodes):
            # Reset the environments
            obs, _ = test_envs.reset(seed=self.seed)
            episode_done = torch.zeros(n_envs, dtype=torch.bool, device=self.device)

            while not episode_done.all():
                with torch.no_grad():
                    action, _, _ = self.sample_action(
                        torch.as_tensor(obs, device=self.device)
                    )
                obs, reward, termination, truncation, _ = test_envs.step(
                    action.cpu().numpy()
                )
                step_done = np.logical_or(termination, truncation)

                # Update total rewards and episode counts using tensor operations
                reward = torch.as_tensor(reward, device=self.device)
                step_done = torch.as_tensor(step_done, device=self.device)

                episode_done |= step_done
                episode_returns[~episode_done, ep_idx] += reward[~episode_done]

        print(episode_returns)
        # Calculate average returns for each environment
        average_return = episode_returns.mean(dim=1)
        return average_return


def make_env_fn(
    seed: int,
    log_dir: Union[str, None] = None,
    capture_video: bool = False,
    render_mode=None,
):
    def single_env(env_idx: int):
        env = gymz.make("Pendulum-v1", render_mode=render_mode)

        if hasattr(env, "action_space"):
            env.action_space.seed(seed)

        # Video capture
        if capture_video and env_idx == 0:
            env = gymz.wrappers.RecordVideo(
                env, log_dir, lambda ep_idx: True if ep_idx == 0 else False
            )

        env = gymz.wrappers.RecordEpisodeStatistics(env)
        # env = gymz.wrappers.ClipAction(env)
        # env = gymz.wrappers.NormalizeObservation(env)
        # env = gymz.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gymz.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gymz.wrappers.TransformReward(
        #     env, lambda reward: np.clip(reward, -10, 10)
        # )
        return env

    return single_env


def main():
    from gymnasium.vector import SyncVectorEnv
    from ..utils import set_torch_seed
    from datetime import datetime

    # Hyperparameters
    # Best found so far (based on HuggingFace)
    n_envs = 4
    total_timesteps = int(1e5)
    batch_size = 256
    seed = 1
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-3  # Example learning rate
    rollout_length = 1024
    gamma = 0.99  # Not hugging fac
    gae_lambda = 0.95
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5
    clip_coef = 0.2
    clip_vloss = False
    norm_adv = False
    target_kl = 0.01
    n_epochs = 10
    anneal_lr = True
    expanse_dim = 256

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"src/checkpoints/ppo-unit-test_{current_datetime}"
    writer = SummaryWriter(log_dir=log_dir)

    # Environment setup
    env_func = make_env_fn(
        seed=seed, log_dir=log_dir, capture_video=False, render_mode=None
    )
    envs = SyncVectorEnv([lambda k=k: env_func(k) for k in range(n_envs)])

    set_torch_seed(seed)

    # Instantiate the PPOAgent
    agent = PPOAgent(
        envs=envs,
        learning_rate=learning_rate,
        device=device,
        writer=writer,
        log_freq=1000,  # Log every 1000 steps
        expanse_dim=expanse_dim,
        seed=seed,
    )

    # Train agent
    agent.train(
        total_timesteps=total_timesteps,
        rollout_length=rollout_length,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        clip_coef=clip_coef,
        clip_vloss=clip_vloss,
        norm_adv=norm_adv,
        target_kl=target_kl,
        batch_size=batch_size,
        n_epochs=n_epochs,
        anneal_lr=anneal_lr,
    )

    # Test and Record Video
    test_env_fn = make_env_fn(
        seed=seed, log_dir=log_dir, capture_video=True, render_mode="rgb_array"
    )
    test_envs = SyncVectorEnv([lambda k=k: test_env_fn(k) for k in range(n_envs)])

    average_returns = agent.evaluate(n_episodes=10, test_envs=test_envs)
    print("Testing - Average Return : \n", average_returns.mean().item())

if __name__ == "__main__":
    main()
