import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchrl.data import ReplayBuffer, ListStorage
from ..assets.models import Actor, SoftQNetwork
from .core import GenericAgent


class CoreticAgent(GenericAgent):
    """Contextual Representation Learning via Time Series Transformers for Control"

    """
    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        repr_model: nn.Module,
        repr_model_learning_rate: float,
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
        self.replay_buffer = ReplayBuffer(storage=ListStorage(buffer_size))
        # self.replay_buffer = ReplayBuffer(
        #     buffer_size,
        #     envs.single_observation_space,
        #     envs.single_action_space,
        #     device,
        #     handle_timeout_termination=True,
        # )

        self.device = device
        self.critic_learning_rate = critic_learning_rate
        self.envs = envs
        # TODO: tensorboard logging
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
        # When using vectorized environment, the environments are automatically reset
        # at the end of an episode and real terminal observation is in infos.
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
            actions, _, _ = self.sample_action(obs, to_numpy=True)

            # print(f"Actions in train, shape: {actions.shape}, type: {type(actions)}")
            next_obs, rewards, dones, infos = self.envs.step(actions)

            # print(f"Observations from env in train, shape: {next_obs.shape}, type: {type(next_obs)}")

            experience = (obs, next_obs, actions, rewards, dones, infos)
            experience = self.preprocess_experience(experience)
            self.update_agent(experience)

            obs = next_obs
            self.global_step += 1
    
    def gather_experience(self, total_timesteps):
        obs = self.envs.reset()
        episode_rewards = []
        
        for _ in range(total_timesteps):
            actions, _, _ = self.sample_action(obs, to_numpy=True)
            next_obs, rewards, dones, infos = self.envs.step(actions)

            # Store the experience in the RolloutBuffer
            self.rollout_buffer.add(obs, actions, rewards, dones, next_obs)

            obs = next_obs
            episode_rewards.append(rewards)

            # If done, preprocess and update agent
            if all(dones):
                experience = self.rollout_buffer.get()
                experience = self.preprocess_experience(experience)
                self.update_agent(experience)
                self.rollout_buffer.reset()

                obs = self.envs.reset()  # Reset environments for the new episode
                episode_rewards = []

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
