import os
import time
from pathlib import PurePath
from typing import Union

import gym
import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from ..assets.models import Actor, SoftQNetwork
from ..meters.episodic import TrackEpisodes
from ..utils import get_action_dim, is_vector_env
from .core import GenericAgent
from ..data.buffer import EpisodicBuffer, EpisodicBufferSamples
from .adaptran import get_observation_dim

class AdaptranV2(GenericAgent):
    def __init__(
        self,
        envs: gymnasium.vector.VectorEnv,  # | gymnasium.Env,
        critic_learning_rate: float,
        actor_learning_rate: float,
        buffer_size: int,
        device: torch.device,
        writer: Union[SummaryWriter, None] = None,
        log_freq: int = 100,
        expanse_dim: int = 256,
        seed: int = 1,
    ):
        super().__init__(envs)
        self.device = device
        self.critic_learning_rate = critic_learning_rate
        self.writer = writer
        self.log_freq = log_freq
        self.seed = seed
        
        self.is_vector_env: bool = is_vector_env(envs)
        assert self.is_vector_env, "Environment not vectorized"

        self.is_contextual_env: bool = getattr(self.envs, "is_contextual_env", False)
        assert self.is_contextual_env, "Only contextual environments are compatible."

        self.single_observation_space = self.envs.single_observation_space
        self.single_action_space = self.envs.single_action_space
        self.n_envs = self.envs.num_envs

        assert isinstance(self.single_action_space, gymnasium.spaces.Box), ValueError(
            f"only continuous action space is supported, given {self.single_action_space}"
        )

        self.single_observation_space["obs"].dtype = np.float32
        self.single_observation_space["context"].dtype = np.float32

        self.action_dim = get_action_dim(self.envs)
        obs_context_dim: dict = get_observation_dim(self.envs)
        self.observation_dim, self.context_dim = (
            obs_context_dim["obs"],
            obs_context_dim["context"],
        )
        self.state_dim = self.observation_dim + self.context_dim

        # Define actor, critic and target networks
        self.actor = Actor(self.envs, self.state_dim, expanse_dim).to(device)
        self.qf1 = SoftQNetwork(self.envs, self.state_dim, expanse_dim).to(device)
        self.qf2 = SoftQNetwork(self.envs, self.state_dim, expanse_dim).to(device)
        self.qf1_target = SoftQNetwork(self.envs, self.state_dim, expanse_dim).to(
            device
        )
        self.qf2_target = SoftQNetwork(self.envs, self.state_dim, expanse_dim).to(
            device
        )
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # Define optimizers for actor and critic
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=critic_learning_rate,
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=actor_learning_rate
        )

        # Define replay buffer
        self.replay_buffer = EpisodicBuffer(
            buffer_size=buffer_size,
            observation_space=envs.single_observation_space,
            action_space=envs.single_action_space,
            device=device,
            n_envs=self.n_envs,
            include_context=self.is_contextual_env,
        )

        self.ep_tracker = TrackEpisodes(writer=self.writer, num_envs=self.n_envs)

    def initialize(
        self,
        batch_size: int,
        learning_starts: int,
        alpha: Union[float, None],
        autotune: bool,
        gamma: float,
        policy_frequency: int,
        target_network_frequency: int,
        tau: float,
    ):
        """
        Initialize a single training run. Sets up essential hyperparameters and configurations.

        Parameters:
            batch_size (int): Size of the minibatch.
            learning_starts (int): Number of environment steps to collect before training starts.
            alpha (Union[float, None]): Scaling factor for the entropy term in the objective. If None,
                autotune is expected to be true and alpha learned automatically.
            autotune (bool): Whether to automatically tune the entropy scaling factor `alpha`.
            gamma (float): Discount factor for future rewards.
            policy_frequency (int): Frequency with which to update the policy network.
            target_network_frequency (int): Frequency with which to update the target network.
            tau (float): Soft update factor.
        """

        self.autotune = autotune

        assert not (alpha is None and not autotune), ValueError(
            "If alpha is not given, autotune has to be enabled."
        )

        # If automatic tuning of alpha is enabled, set up the corresponding variables and
        # optimizers
        if autotune:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.single_action_space.shape).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=self.critic_learning_rate
            )
        else:
            self.alpha = alpha

        # Initialize various training hyperparameters
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.gamma = gamma
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau

        # Initialize the global step counter
        self.global_step = 0

    def sample_action(self, observations):
        """
        Sample an action from the policy given the current observations.

        Args:
            observations: The current state observations from the environment.
            to_numpy (bool): Flag indicating whether to convert the tensor outputs to
                NumPy arrays. Useful when sampled actions are stored in replay buffer.

        Returns:
            actions: Actions sampled from the current policy.
            log_probs: Log probabilities of the sampled actions.
            squashed_means: Squashed mean values of the action distributions.

        Note: log_probs and squashed_means will be None if the global step is less than
            learning_starts.
        """
        actions = log_probs = squashed_means = None

        # Check if the agent should start learning
        if self.global_step < self.learning_starts:
            # Randomly sample actions if learning has not started
            actions = np.array(
                [self.single_action_space.sample() for _ in range(self.n_envs)]
            )
            # actions = self.envs.unwrapped.action_space.sample()
            return actions, log_probs, squashed_means

        # Convert observations to Tensor if they're not already
        observations = torch.as_tensor(
            observations, device=self.device, dtype=torch.float32
        )

        # Use the actor model to sample actions, log_probs, and squashed_means
        actions, log_probs, squashed_means = self.actor.get_action(observations)
        return actions, log_probs, squashed_means

    def preprocess_experience(self, obs, next_obs, actions, rewards, dones, infos, context):
        """Preprocess experience if needed (e.g., stacking frames, normalizing)."""

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        # When using vectorized environment, the environments are automatically reset
        # at the end of an episode and real terminal observation is in infos.
        # For more info: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
        real_next_obs = next_obs.copy()
        # next_obs = self.preprocess_observation(real_next_obs)

        for idx, done in enumerate(dones):
            if done:  # if the sub-environment has terminated
                real_next_obs[idx] = infos["final_observation"][idx]["obs"]

        # Add any preprocessing code here
        return obs, real_next_obs, actions, rewards, dones, infos, context

    def unpack_observation(self, obs):
        if not isinstance(obs, dict):
            ValueError(
                "Contextual environments as expected to return dictionary observations."
            )
        return obs["obs"], obs["context"]

    def update_agent(self, experience):
        """
        Update the agent's models (Q-networks, policy, value function, etc.) based on experience.

        Args:
            experience: A tuple containing the data for one timestep (or episode),
                usually (obs, next_obs, actions, rewards, dones, infos).

        Steps involved:
            1. Add experience to the replay buffer.
            2. If enough samples are gathered, perform updates:
                - Update critic network based sampled experience from replay buffer.
                - Update actor network also based on sampled experience.
                - Update target networks if it's time to do so.
        """
        # Add new experience to the episodic replay buffer
        # Assuming experience is a tuple: (obs, real_next_obs, actions, rewards, dones, infos)

        self.replay_buffer.add(*experience)

        if self.global_step < self.learning_starts:
            return

        # Sample batch of transitions from replay buffer
        batch: EpisodicBufferSamples = self.replay_buffer.sample(self.batch_size, desired_length=1)

        if batch.observations.shape[1] == 1:
            batch.observations = batch.observations.squeeze(1)
            batch.next_observations = batch.next_observations.squeeze(1)
            batch.actions = batch.actions.squeeze(1)
            batch.rewards = batch.rewards.squeeze(1)
            batch.dones = batch.dones.squeeze(1)
            batch.contexts = batch.contexts.squeeze(1)

        # Update the Critic network
        self.update_critic(batch)

        # Update the Actor network
        # Only update the actor network periodically, as defined by policy_frequency
        if self.global_step % self.policy_frequency == 0:
            self.update_actor(batch)

        # Update the Target Networks
        # Only update the target networks periodically, as defined by target_network_frequency
        if self.global_step % self.target_network_frequency == 0:
            self.update_target_networks()

    def update_critic(self, batch: EpisodicBufferSamples):
        """
        Updates the critic network using the given samples and state sequences.

        Args:
            batch (ReplayBufferSamples): The batch of transitions sampled from the
                replay buffer. Each sequence contains observations, actions, rewards,
                and done flags.
        Returns:
            None: The method updates the critic network in-place.
        """

        states: torch.Tensor = torch.cat([
            batch.observations,
            batch.contexts
        ], dim=-1)

        next_states: torch.Tensor = torch.cat([
            batch.next_observations,
            batch.contexts
        ], dim=-1)

        # states, next_states = states.squeeze(1), next_states.squeeze(1)

        with torch.no_grad():
            # 1. Sample next action and compute Q-value targets
            next_state_actions, next_state_log_pi, _ = self.sample_action(
                next_states
            )
            qf1_next_target = self.qf1_target(
                next_states, next_state_actions
            )
            qf2_next_target = self.qf2_target(
                next_states, next_state_actions
            )

            # 2. Compute the minimum Q-value target and the target for the Q-function
            # update
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = batch.rewards.flatten() + (
                1 - batch.dones.flatten()
            ) * self.gamma * min_qf_next_target.view(-1)

        # 3. Compute the Q-values and the MSE loss for both critics
        qf1_a_values = self.qf1(states, batch.actions).view(-1)
        qf2_a_values = self.qf2(states, batch.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # 4. Update critic model
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if self.global_step % self.log_freq == 0 and self.writer:
            self.writer.add_scalar(
                "losses/qf1_values", qf1_a_values.mean().item(), self.global_step
            )
            self.writer.add_scalar(
                "losses/qf2_values", qf2_a_values.mean().item(), self.global_step
            )
            self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), self.global_step)
            self.writer.add_scalar("losses/qf2_loss", qf2_loss.item(), self.global_step)
            self.writer.add_scalar(
                "losses/qf_loss", qf_loss.item() / 2.0, self.global_step
            )

    def update_actor(self, batch: EpisodicBufferSamples):
        """
        Updates the actor network using the given state sequences.

        Args:
            batch (ReplayBufferSamples): The batch of transitions sampled from the
                replay buffer. Each sequence contains observations, actions, rewards,
                and done flags.

        Returns:
            None: The method updates the actor network and optionally the
                temperature parameter in-place.
        """
        # The loop here runs multiple updates for each actor update,
        # which is specified by self.policy_frequency.

        states: torch.Tensor = torch.cat([
            batch.observations,
            batch.contexts
        ], dim=-1)
        
        for _ in range(self.policy_frequency):
            # 1. Sample Actions:
            # Using the current policy, sample actions and their log
            # probabilities from the current state batch.
            pi, log_pi, _ = self.sample_action(states)

            # 2. Compute the Q-values of the sampled actions using both the
            # Q-functions (Q1 and Q2).
            qf1_pi = self.qf1(states, pi)
            qf2_pi = self.qf2(states, pi)

            # Take the minimum Q-value among the two Q-functions to improve
            # robustness.
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            # 3. Compute Actor Loss:
            # The actor aims to maximize this quantity, which corresponds
            # to maximizing Q-value and entropy.
            actor_loss = (self.alpha * log_pi - min_qf_pi).mean()

            # 4. Perform backpropagation to update the actor network.
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 5. Update Temperature (Optional):
            # If self.autotune is True, the temperature parameter alpha is
            # also learned.
            if self.autotune:
                # Sample actions again (not strictly needed, could reuse above)
                with torch.no_grad():
                    _, log_pi, _ = self.sample_action(states)

                # Compute the loss for alpha, aiming to keep policy entropy
                # close to target_entropy.
                alpha_loss = (
                    -self.log_alpha.exp() * (log_pi + self.target_entropy)
                ).mean()

                # Perform backpropagation to update alpha.
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                # Update the alpha value.
                self.alpha = self.log_alpha.exp().item()

        # Log Actor loss and alpha every self.log_freq steps.
        if self.global_step % self.log_freq == 0 and self.writer:
            self.writer.add_scalar(
                "losses/actor_loss", actor_loss.item(), self.global_step
            )
            self.writer.add_scalar("alpha", self.alpha, self.global_step)

            if self.autotune:
                self.writer.add_scalar(
                    "losses/alpha_loss", alpha_loss.item(), self.global_step
                )

    def update_target_networks(self):
        for param, target_param in zip(
            self.qf1.parameters(), self.qf1_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(
            self.qf2.parameters(), self.qf2_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def train(
        self,
        total_timesteps: int,
        batch_size: int,
        learning_starts: int,
        alpha: Union[float, None],
        autotune: bool,
        gamma: float,
        policy_frequency: int,
        target_network_frequency: int,
        tau: float,
    ):
        """
        Train the Soft Actor-Critic (SAC) agent.

        Args:
            total_timesteps (int): The total number of timesteps to train the agent for.
            batch_size (int): The size of each batch of experiences used for training.
            learning_starts (int): The timestep at which learning should begin.
            alpha (Union[float, None]): The temperature parameter for the SAC algorithm.
                                    If None, it will be learned if autotune is True.
            autotune (bool): Whether to automatically tune the temperature parameter.
            gamma (float): The discount factor for future rewards.
            policy_frequency (int): The frequency with which the policy should be updated.
            target_network_frequency (int): The frequency of updating the target network.
            tau (float): The soft update coefficient for updating the target network.
        """
        # Initialize the SAC agent with the provided parameters
        self.initialize(
            batch_size=batch_size,
            learning_starts=learning_starts,
            alpha=alpha,
            autotune=autotune,
            gamma=gamma,
            policy_frequency=policy_frequency,
            target_network_frequency=target_network_frequency,
            tau=tau,
        )

        # Start timer
        start_time = time.time()

        # Reset the environment and get initial observation
        curr_obs, _ = self.envs.reset(seed=self.seed)
        curr_obs, curr_context = self.unpack_observation(curr_obs)

        for _ in range(total_timesteps):
            # Sample actions
            with torch.no_grad():
                state = np.concatenate([curr_obs, curr_context], axis=-1)
                actions, _, _ = self.sample_action(state)  # (n_envs, action_dim)

            actions = actions.detach().cpu().numpy() if torch.is_tensor(actions) else actions

            # Execute actions in the
            next_obs, rewards, dones, infos = self.envs.step(actions)
            self.ep_tracker.step(
                global_step=self.global_step, rewards=rewards, dones=dones
            )

            next_obs, next_context = self.unpack_observation(next_obs)

            # Prepare experience for the agent's update
            experience = self.preprocess_experience(
                curr_obs, next_obs, actions, rewards, dones, infos, curr_context
            )

            # Update the agent
            self.update_agent(experience)

            assert np.array_equal(
                curr_context, next_context
            ), "Non-stationary environment"

            # Update the current observation
            curr_obs = next_obs

            curr_context = next_context
            # Log episodic information if available
            # Episodic information from any one of the environments is sufficient
            # TODO: Uncomment logging if TrackEpisode not present
            # if "episode" in infos or np.any(dones):
            #     done_idx = np.argmax(dones)
            #     print(
            #         f"global_step={self.global_step}, episodic_return={infos[done_idx]['episode']['r'].item()}"
            #     )

            #     if self.writer:
            #         self.writer.add_scalar(
            #             "train/episodic_return",
            #             infos[done_idx]["episode"]["r"].item(),
            #             self.global_step,
            #         )
            #         self.writer.add_scalar(
            #             "train/episodic_length",
            #             infos[done_idx]["episode"]["l"].item(),
            #             self.global_step,
            #         )

            # Log steps per second (SPS) every self.log_freq steps
            if self.global_step % self.log_freq == 0:
                print("SPS:", int(self.global_step / (time.time() - start_time)))
                if self.writer:
                    self.writer.add_scalar(
                        "train/SPS",
                        int(self.global_step / (time.time() - start_time)),
                        self.global_step,
                    )

            # Increment the global step count
            self.global_step += 1

    def test(self, n_episodes=10):
        """
        Test the trained Soft Actor-Critic (SAC) agent (including vectorized environment).

        Args:
            n_episodes (int): The number of episodes to run for testing per environment.

        Returns:
            None
        """
        # Initialize a list to keep track of all returns for each environment
        # and number of episodes progresed in each of them.
        all_returns = [[] for _ in range(self.n_envs)]
        episode_count = [0] * self.n_envs

        # Initialize a list to keep track of returns for ongoing episode in each environment
        episodic_returns = [0] * self.envs.num_envs

        # Reset the environments to get an initial observation state
        curr_obs, _ = self.envs.reset()
        if self.is_contextual_env:
            curr_obs, _ = curr_obs["obs"], curr_obs["context"]

        while min(episode_count) < n_episodes:
            # Sample actions from the trained policy
            actions, _, _ = self.sample_action(curr_obs, to_numpy=True)

            # Execute the actions in the environments
            next_obs, rewards, dones, _ = self.envs.step(actions)
            if self.is_contextual_env:
                next_obs, _ = next_obs["obs"], next_obs["context"]
            # Update the episode rewards
            for i, (reward, done) in enumerate(zip(rewards, dones)):
                episodic_returns[i] += reward

                if done and episode_count[i] < n_episodes:
                    print(
                        f"Environment {i+1}, Episode {episode_count[i] + 1}: Total Reward: {episodic_returns[i]}"
                    )
                    if self.writer:
                        self.writer.add_scalar(
                            f"test/env_{i+1}/episodic_return",
                            episodic_returns[i],
                            episode_count[i] + 1,
                        )
                    all_returns[i].append(episodic_returns[i])

                    # Reset episodic return and increment count
                    episodic_returns[i] = 0
                    episode_count[i] += 1

            # Update the current observations
            curr_obs = next_obs

        # Calculate and log the average returns over all test episodes for each environment
        for i, returns in enumerate(all_returns):
            avg_return = sum(returns) / len(returns)
            print(
                f"Environment {i+1}: Average Return over {n_episodes} episodes: {avg_return}"
            )
            if self.writer:
                self.writer.add_scalar(f"test/avg_return", avg_return, i + 1)

    def save_checkpoint(self, checkpoint_dir: PurePath):
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = checkpoint_dir / "checkpoint.pth"
        checkpoint = {"actor_state_dict": self.actor.state_dict()}

        torch.save(checkpoint, str(checkpoint_path))


if __name__ == "__main__":
    # Hyperparameters
    # TODO: Update unit test
    exp_name = "sac_experiment"
    env_id = "Hopper-v4"
    total_timesteps = 1000000
    buffer_size = int(1e6)
    gamma = 0.99
    tau = 0.005
    batch_size = 256
    learning_starts = 5e3
    policy_lr = 3e-4
    q_lr = 1e-3
    policy_frequency = 2
    target_network_frequency = 1
    noise_clip = 0.5
    alpha_value = 0.2
    autotune = True
    seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    envs = gymnasium.make_env(env_id, seed)

    agent = SACAgent()
