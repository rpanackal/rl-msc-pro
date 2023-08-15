import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from .core import GenericAgent

import numpy as np

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        """Initialize the Soft Q-Network.

        Args:
            env: Gym environment. Used to get the observation and action dimensions.
        """
        super().__init__()

        # Calculate total input size from observation and action space
        input_dim = np.prod(env.single_observation_space.shape) + np.prod(
            env.single_action_space.shape
        )

        # Define the neural network layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        """Pass the observation and action through the neural network to get the Q-value.

        Args:
            x: Observation tensor.
            a: Action tensor.

        Returns:
            Q-value for the given observation and action.
        """
        # Concatenate the observation and action to form the input

        # print("Type check in critic forward: ", type(x), type(a))
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        x = torch.cat([x, a], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        """Initialize the Actor model.

        Args:
            env: Gym environment. Used to get the observation and action dimensions.
        """
        super().__init__()
        observation_dim = np.prod(env.single_observation_space.shape)
        action_dim = np.prod(env.single_action_space.shape)

        self.fc1 = nn.Linear(observation_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)

        # Action rescaling parameters, to ensure actions are within bounds
        action_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_bias = (env.action_space.high + env.action_space.low) / 2.0

        self.register_buffer(
            "action_scale", torch.tensor(action_scale, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor(action_bias, dtype=torch.float32)
        )

    def forward(self, x):
        """Pass the observation through the neural network to get mean and
        log_std of the action distribution.

        Args:
            x: Observation tensor.

        Returns:
            mean, log_std: Mean and log standard deviation of the action
                distribution.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = self._squash_log_std(log_std)
        return mean, log_std

    def _squash_log_std(self, log_std):
        """Apply tanh squashing to log_std to ensure it's within bounds.

        The squashing function ensures that log_std stays within [LOG_STD_MIN,
        LOG_STD_MAX]. This is critical for numerical stability. Using a log standard
        deviation instead of the standard deviation allows the network to output
        negative values, making optimization easier.

        Args:
            log_std: Unsquashed log standard deviation tensor.

        Returns:
            Squashed log standard deviation tensor.
        """
        log_std = torch.tanh(log_std)
        return LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    def get_action(self, x):
        """Sample an action from the actor's policy, given an observation.

        Args:
            x: Observation tensor.

        Returns:
            action, log_prob, mean: Sampled action, log probability of the action,
            and mean action.
        """
        mean, log_std = self(x)
        std = log_std.exp()
        action, log_prob, squashed_mean = self._sample_action(mean, std)
        return action, log_prob, squashed_mean

    def _sample_action(self, mean, std):
        """Sample an action using the reparameterization trick, applying tanh
        squashing.

        Args:
            mean: Mean of the action distribution.
            std: Standard deviation of the action distribution.

        Returns:
            action, log_prob, squashed_mean: Sampled action, log probability of
            the action, and mean action after squashing.

        The reparameterization trick allows gradients to flow through the stochastic
        sampling. This is essential for learning the optimal policy.

        Tanh squashing ensures that actions are bounded within the environment's
        limits.

        The action scaling and bias ensure that actions are appropriately scaled
        and centered within the environment's bounds.
        """
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)  # Tanh squashing to bound the actions

        # Apply action scaling and bias
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforce action bound; this correction term is necessary when using tanh
        # squashing
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        squashed_mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, squashed_mean


class SACAgent(GenericAgent):
    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        critic_learning_rate: float,
        actor_learning_rate: float,
        buffer_size: int,
        device: torch.device,
        # writer,
    ):
        assert isinstance(
            envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"

        self.actor = Actor(envs).to(device)
        self.qf1 = SoftQNetwork(envs).to(device)
        self.qf2 = SoftQNetwork(envs).to(device)
        self.qf1_target = SoftQNetwork(envs).to(device)
        self.qf2_target = SoftQNetwork(envs).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=critic_learning_rate,
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=actor_learning_rate
        )

        envs.single_observation_space.dtype = np.float32
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=True,
        )

        self.device = device
        self.critic_learning_rate = critic_learning_rate
        self.envs = envs
        # self.writer = writer

        super().__init__(envs)

    def initialize(
        self,
        batch_size: int,
        learning_starts: int,
        alpha: float | None,
        autotune: bool,
        gamma: float,
        policy_frequency: int,
        target_network_frequency: int,
        tau: float,
    ):
        """Initialize a single training run"""

        self.autotune = autotune
        assert alpha != None and autotune
        if autotune:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.envs.single_action_space.shape).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=self.critic_learning_rate
            )
        else:
            self.alpha = alpha

        self.batch_size = batch_size
        self.learning_starts = learning_starts

        self.gamma = gamma
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau

        self.global_step = 0

    def sample_action(self, observations, to_numpy=False):
        """Sample action from the policy given current observation."""
        if self.global_step < self.learning_starts:
            actions = np.array(
                [
                    self.envs.single_action_space.sample()
                    for _ in range(self.envs.num_envs)
                ]
            )
            log_probs = None
            squashed_means = None
        else:
            if not isinstance(observations, torch.Tensor):
                observations = torch.Tensor(observations).to(self.device)
            actions, log_probs, squashed_means = self.actor.get_action(observations)

            if to_numpy:
                return actions.detach().numpy(), log_probs.detach().numpy(), squashed_means.detach().numpy()

        return actions, log_probs, squashed_means

    def preprocess_experience(self, experience):
        """Preprocess experience if needed (e.g., stacking frames, normalizing)."""
        # Assuming experience is a tuple: (obs, next_obs, actions, rewards, dones, infos)
        obs, next_obs, actions, rewards, dones, infos = experience

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        # Add any preprocessing code here
        return obs, real_next_obs, actions, rewards, dones, infos

    def update_agent(self, experience):
        """Update the agent (e.g., Q-network, policy, value function) based on experience."""
        # Assuming experience is a tuple: (obs, real_next_obs, actions, rewards, dones, infos)
        self.replay_buffer.add(*experience)
        if self.global_step < self.learning_starts:
            return

        # Sample from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)

        self.update_critic(batch)

        if self.global_step % self.policy_frequency == 0:
            self.update_actor(batch)

        if self.global_step % self.target_network_frequency == 0:
            self.update_target_networks()

    def update_critic(self, batch: ReplayBufferSamples):
        # Compute Critic loss
        # print(f"Obsevations in batch, shape: {batch.next_observations.shape}, type: {type(batch.next_observations)}")
        # print(f"Actions in train, shape: {batch.actions.shape}, type: {type(batch.actions)}")
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.sample_action(
                batch.next_observations
            )

            qf1_next_target = self.qf1_target(
                batch.next_observations, next_state_actions
            )
            qf2_next_target = self.qf2_target(
                batch.next_observations, next_state_actions
            )

            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = batch.rewards.flatten() + (
                1 - batch.dones.flatten()
            ) * self.gamma * min_qf_next_target.view(-1)

        qf1_a_values = self.qf1(batch.observations, batch.actions).view(-1)
        qf2_a_values = self.qf2(batch.observations, batch.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # Update critic model
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

    def update_actor(self, batch: ReplayBufferSamples):
        # Compute Actor loss

        for _ in range(self.policy_frequency):
            pi, log_pi, _ = self.sample_action(batch.observations)
            qf1_pi = self.qf1(batch.observations, pi)
            qf2_pi = self.qf2(batch.observations, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
            actor_loss = (self.alpha * log_pi - min_qf_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.autotune:
                with torch.no_grad():
                    _, log_pi, _ = self.sample_action(batch.observations)
                alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()

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
        alpha: float | None,
        autotune: bool,
        gamma: float,
        policy_frequency: int,
        target_network_frequency: int,
        tau: float,
    ):
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

        obs = self.envs.reset()
        for _ in range(total_timesteps):
            # ALGO LOGIC: put action logic here
            # print("Training step: ", self.global_step)
            actions, _, _ = self.sample_action(obs, to_numpy=True)

            # print(f"Actions in train, shape: {actions.shape}, type: {type(actions)}")
            next_obs, rewards, dones, infos = self.envs.step(actions)

            # print(f"Observations from env in train, shape: {next_obs.shape}, type: {type(next_obs)}")

            experience = (obs, next_obs, actions, rewards, dones, infos)
            experience = self.preprocess_experience(experience)
            self.update_agent(experience)

            obs = next_obs
            self.global_step += 1


if __name__ == "__main__":
    # Hyperparameters
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

    envs = gym.make_env(env_id, seed)

    agent = SACAgent()
