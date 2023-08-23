import time

import gym
import numpy as np
import torch
import torch.nn.functional as F
from gym.vector import VectorEnv
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from ..assets import Autoformer
from ..assets.models import Actor, SoftQNetwork
from ..data.buffer import EpisodicBuffer, EpisodicBufferSamples
from .core import GenericAgent


class CoreticAgent(GenericAgent):
    """Contextual Representation Learning via Time Series Transformers for Control" """

    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        repr_model: Autoformer,
        repr_model_learning_rate: float,
        critic_learning_rate: float,
        actor_learning_rate: float,
        buffer_size: int,
        device: torch.device,
        writer: SummaryWriter,
        log_freq: int = 100,
    ):
        assert isinstance(envs.single_action_space, gym.spaces.Box), ValueError(
            "only continuous action space is supported"
        )

        self.repr_model = repr_model
        self.src_seq_length = self.repr_model.src_seq_length
        self.tgt_seq_length = self.repr_model.tgt_seq_length
        self.repr_dim = self.repr_model.embed_dim

        self.actor = Actor(envs, self.repr_dim).to(device)
        self.qf1 = SoftQNetwork(envs, self.repr_dim).to(device)
        self.qf2 = SoftQNetwork(envs, self.repr_dim).to(device)
        self.qf1_target = SoftQNetwork(envs, self.repr_dim).to(device)
        self.qf2_target = SoftQNetwork(envs, self.repr_dim).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.repr_optimizer = optim.Adam(
            list(self.repr_model.parameters()),
            lr=repr_model_learning_rate,
        )
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=critic_learning_rate,
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=actor_learning_rate
        )

        envs.single_observation_space.dtype = np.float32
        self.n_envs = envs.num_envs if isinstance(envs, VectorEnv) else 1
        self.replay_buffer = EpisodicBuffer(
            buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            n_envs=self.n_envs,
            handle_timeout_termination=True,
        )

        self.device = device
        self.critic_learning_rate = critic_learning_rate
        self.envs = envs
        self.writer = writer
        self.log_freq = log_freq

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
        trajectory_length: int | None = None,
    ):
        """
        Initialize a single training run. Sets up essential hyperparameters and configurations.

        Parameters:
            batch_size (int): Size of the minibatch.
            learning_starts (int): Number of environment steps to collect before training starts.
            alpha (float | None): Scaling factor for the entropy term in the objective. If None,
                autotune is expected to be true and alpha learned automatically.
            autotune (bool): Whether to automatically tune the entropy scaling factor `alpha`.
            gamma (float): Discount factor for future rewards.
            policy_frequency (int): Frequency with which to update the policy network.
            target_network_frequency (int): Frequency with which to update the target network.
            tau (float): Soft update factor.
            trajectory_length (int | None): Desired length of trajectory samples from episodic replay
                buffer.

        Raises:
            ValueError: If `trajectory_length` is too small for the given source and target lengths.
        """

        self.autotune = autotune

        assert not (alpha is None and not autotune), ValueError(
            "If alpha is not given, autotune has to be enabled."
        )

        # If automatic tuning of alpha is enabled, set up the corresponding variables and optimizers
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

        # Initialize various training hyperparameters
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.gamma = gamma
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau

        # Set the trajectory length, falling back to the sum of source and target lengths if
        # not specified
        self.trajectory_length = (
            trajectory_length
            if trajectory_length
            else self.src_seq_length + self.tgt_seq_length
        )

        assert (
            self.trajectory_length >= self.src_seq_length + self.tgt_seq_length
        ), ValueError(
            f"Trajectory length({trajectory_length}) too small to for given source length ({trajectory_length})\
            and target length ({trajectory_length})"
        )

        # Initialize the global step counter
        self.global_step = 0

    def sample_action(self, observations, to_numpy=False):
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
        # Check if the agent should start learning
        if self.global_step < self.learning_starts:
            # Randomly sample actions if learning has not started
            actions = np.array(
                [self.envs.single_action_space.sample() for _ in range(self.n_envs)]
            )
            log_probs = None
            squashed_means = None
        else:
            # Convert observations to Tensor if they're not already
            if not isinstance(observations, torch.Tensor):
                observations = torch.Tensor(observations).to(self.device)

            # Use the actor model to sample actions, log_probs, and squashed_means
            actions, log_probs, squashed_means = self.actor.get_action(observations)

            # Convert Tensor to NumPy array if to_numpy is True.
            if to_numpy:
                return (
                    actions.detach().cpu().numpy(),
                    log_probs.detach().cpu().numpy(),
                    squashed_means.detach().cpu().numpy(),
                )

        return actions, log_probs, squashed_means

    def preprocess_experience(self, experience):
        """Preprocess experience if needed (e.g., stacking frames, normalizing)."""
        # Assuming experience is a tuple: (obs, next_obs, actions, rewards, dones, infos)
        obs, next_obs, actions, rewards, dones, infos = experience

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        # When using vectorized environment, the environments are automatically reset
        # at the end of an episode and real terminal observation is in infos.
        # For more info: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        # Add any preprocessing code here
        return obs, real_next_obs, actions, rewards, dones, infos

    def update_agent(self, experience):
        """
        Update the agent's models (e.g., Representation model, Q-networks, policy, value function)
            based on episodic experience.

        Args:
            experience: A tuple containing one timestep's worth of data,
                usually (obs, next_obs, actions, rewards, dones, infos).

        Steps involved:
            1. Add experience to the episodic replay buffer.
            2. If enough samples are gathered, perform updates:
                - Sample random episodes from the episodic replay buffer.
                - Update representation model to get latent state sequences.
                - Update critic network based on these latent states and other experience.
                - Periodically update the actor and target networks.

        Note: Unlike traditional replay buffers, this uses an episodic replay buffer that
        samples entire episodes for training.
        """
        # Add new experience to the episodic replay buffer
        # Assuming experience is a tuple: (obs, real_next_obs, actions, rewards, dones, infos)
        self.replay_buffer.add(*experience)

        if self.global_step < self.learning_starts:
            return

        # Sample a batch of random episodes/trajectories from episodic replay buffer
        samples = self.replay_buffer.sample(self.batch_size, self.trajectory_length)

        # Update Represenation Model
        # This will provide a compact state representation for each observation
        state_sequences = self.update_repr_model(samples)

        # Update the Critic network
        self.update_critic(samples, state_sequences)

        # Update the Actor network
        # Only update the actor network periodically, as defined by policy_frequency
        if self.global_step % self.policy_frequency == 0:
            self.update_actor(state_sequences)

        # Update the Target Networks
        # Only update the target networks periodically, as defined by target_network_frequency
        if self.global_step % self.target_network_frequency == 0:
            self.update_target_networks()

    def update_repr_model(self, samples: EpisodicBufferSamples):
        """
        Updates the state representation learning model using the provided batch of
        experience.

        Args:
            samples (EpisodicBufferSamples):  The batch of experience sequences sampled
                from the replay buffer. Each sequence contains observations, actions, rewards,
                and done flags.

        Returns:
            state_sequences (torch.Tensor): The state representation used for critic
                and actor updates.
                shape: (batch_size, state_seq_len, repr_dim).
        """
        # Calculate the number of subsequences as source to produce states per trajectory
        state_seq_len = self.trajectory_length - self.tgt_seq_length

        state_sequences = torch.empty(
            (self.batch_size, state_seq_len, self.repr_dim)
        ).to(self.device)
        total_loss = 0

        # Loop over each potential starting point for the source sequence within a trajectory.
        for t in range(state_seq_len):
            # 1. Calculate the bounds for the source and target sequences
            src_start = max(0, t - self.src_seq_length + 1)
            tgt_end = t + 1 + self.tgt_seq_length  # exclusive

            # 2. Split along sequence dimension into source and target sequences
            src, tgt = (
                samples.observations[:, src_start : t + 1, :],
                samples.observations[:, t + 1, tgt_end, :],
            )

            # Pad source to match expected src_seq_length
            src = self.pad_source(src)

            # 3. Produce state representations and future observations
            predictions, enc_out = self.repr_model(src, full_output=True)

            # 4. Store final element along sequence dimension of enc_out is the state s_t,
            # given obs o_t.
            state_sequences[t] = enc_out[:, -1, :]

            # 5. Compute representation model loss
            total_loss += F.mse_loss(predictions, tgt)

        # 6. Mean loss across all source-target pairs in a trajectory
        total_loss /= state_seq_len

        # 7. Perform backpropagation to update the representation learning model.
        self.repr_optimizer.zero_grad()
        total_loss.backward()
        self.repr_optimizer.step()

        return state_sequences

    def update_critic(self, samples: EpisodicBufferSamples, state_sequences):
        """
        Updates the critic network using the given samples and state sequences.

        Args:
            samples (EpisodicBufferSamples): The batch of experience sequences sampled
                from the replay buffer. Each sequence contains observations, actions, rewards,
                and done flags.
            state_sequences (torch.Tensor): The batch of latent state sequences corresponding
                to the observations in 'samples'.
                shape: (batch_size, state_seq_len, repr_dim).
        """
        # Number of source sequences per trajectory
        state_seq_len = state_sequences.size(1)

        # Initialize loss and current states
        curr_states = state_sequences[:, 0, :]
        qf_loss = 0

        # Loop through each time step in the trajectory
        # Note: We don't have next_states for the last time step in each trajectory,
        # so we loop until the (n_src_per_batch - 1)th time step.
        for t in range(state_seq_len - 1):
            # 1. Get next state sequences
            next_states = state_sequences[:, t + 1, :]

            with torch.no_grad():
                # 2. Sample next action and compute Q-value targets
                next_state_actions, next_state_log_pi, _ = self.sample_action(
                    next_states
                )
                qf1_next_target = self.qf1_target(next_states, next_state_actions)
                qf2_next_target = self.qf2_target(next_states, next_state_actions)

                # 3. Compute the minimum Q-value target and the target for the Q-function update
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - self.alpha * next_state_log_pi
                )
                next_q_value = samples.rewards[:, t].flatten() + (
                    1 - samples.dones[:, t].flatten()
                ) * self.gamma * min_qf_next_target.view(-1)

            # 4. Compute the Q-values and the MSE loss for both critics
            qf1_a_values = self.qf1(curr_states, samples.actions[:, t, :]).view(-1)
            qf2_a_values = self.qf2(curr_states, samples.actions[:, t, :]).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss += qf1_loss + qf2_loss

            # Update current states
            curr_states = next_states

        # Mean loss across all source-target pairs in a trajectory
        qf_loss /= state_seq_len - 1

        # 5. Update critic model
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # Log Q-function loss information every 100 steps.
        if self.global_step % self.log_freq == 0:
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

    def update_actor(self, state_sequences):
        """
        Updates the actor network using the given state sequences.

        Args:
            state_sequences (torch.Tensor): The batch of latent state sequences corresponding
                to the observations.
                shape: (batch_size, state_seq_len, repr_dim).

        Returns:
            None: The method updates the actor network and optionally the temperature parameter
                in-place.
        """

        for _ in range(self.policy_frequency):
            # Number of source sequences per trajectory
            n_src_per_batch = state_sequences.size(1)

            # Initialize actor loss and current batch of states
            curr_states = state_sequences[:, 0, :]
            actor_loss = 0

            for t in range(n_src_per_batch):
                curr_states = state_sequences[:, t, :]

                # 1. Sample Actions:
                # Using the current policy, sample actions and their log
                # probabilities from the current batch of states.
                pi, log_pi, _ = self.sample_action(curr_states)

                # 2. Compute the Q-values of the sampled actions using both the
                # Q-functions (Q1 and Q2).
                qf1_pi = self.qf1(curr_states, pi)
                qf2_pi = self.qf2(curr_states, pi)

                # Take the minimum Q-value among the two Q-functions to improve
                # robustness.
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)

                # 3. Compute Actor Loss:
                # The actor aims to maximize this quantity, which corresponds
                # to maximizing Q-value and entropy.
                actor_loss += (self.alpha * log_pi - min_qf_pi).mean()

            # Mean loss across all source-target pairs in a trajectory
            actor_loss /= n_src_per_batch

            # 4. Perform backpropagation to update the actor network.
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 5. Update Temperature (Optional):
            # If self.autotune is True, the temperature parameter alpha is
            # also learned.
            if self.autotune:
                alpha_loss = 0
                for t in range(n_src_per_batch):
                    curr_states = state_sequences[:, t, :]
                    # Sample actions again (not strictly needed, could reuse above)
                    with torch.no_grad():
                        _, log_pi, _ = self.sample_action(curr_states)

                    # Compute the loss for alpha, aiming to keep policy entropy
                    # close to target_entropy.
                    alpha_loss += (
                        -self.log_alpha * (log_pi + self.target_entropy)
                    ).mean()

                # Mean loss across all source-target pairs in a trajectory
                alpha_loss /= n_src_per_batch

                # Perform backpropagation to update alpha.
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                # Update the alpha value.
                self.alpha = self.log_alpha.exp().item()

        # Log Actor loss and alpha every self.log_freq steps.
        if self.global_step % self.log_freq == 0:
            self.writer.add_scalar(
                "losses/actor_loss", actor_loss.item(), self.global_step
            )
            self.writer.add_scalar("losses/alpha", self.alpha, self.global_step)

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
        alpha: float | None,
        autotune: bool,
        gamma: float,
        policy_frequency: int,
        target_network_frequency: int,
        tau: float,
        trajectory_length: int,
    ):
        """
        Train the Soft Actor-Critic (SAC) agent.

        Args:
            total_timesteps (int): The total number of timesteps to train the agent for.
            batch_size (int): The size of each batch of experiences used for training.
            learning_starts (int): The timestep at which learning should begin.
            alpha (float | None): The temperature parameter for the SAC algorithm.
                If None, it will be learned if autotune is True.
            autotune (bool): Whether to automatically tune the temperature parameter.
            gamma (float): The discount factor for future rewards.
            policy_frequency (int): The frequency with which the policy should be updated.
            target_network_frequency (int): The frequency of updating the target network.
            tau (float): The soft update coefficient for updating the target network.
            trajectory_length (int | None): The length of trajectories sampled from episodic
                replay buffer.
        """
        self.initialize(
            batch_size=batch_size,
            learning_starts=learning_starts,
            alpha=alpha,
            autotune=autotune,
            gamma=gamma,
            policy_frequency=policy_frequency,
            target_network_frequency=target_network_frequency,
            tau=tau,
            trajectory_length=trajectory_length,
        )

        # Start timer
        start_time = time.time()

        # Reset the environment and get initial observation
        obs = self.envs.reset()
        for _ in range(total_timesteps):
            # Sample actions
            actions, _, _ = self.sample_action(obs, to_numpy=True)

            # Execute actions in the environment
            next_obs, rewards, dones, infos = self.envs.step(actions)

            # Prepare experience for the agent's update
            experience = (obs, next_obs, actions, rewards, dones, infos)
            experience = self.preprocess_experience(experience)

            # Update the agent
            self.update_agent(experience)

            # Update the current observation
            obs = next_obs

            # Log episodic information if available
            for info in infos:
                if "episode" in info.keys():
                    print(
                        f"global_step={self.global_step}, episodic_return={info['episode']['r']}"
                    )
                    self.writer.add_scalar(
                        "train/episodic_return", info["episode"]["r"], self.global_step
                    )
                    self.writer.add_scalar(
                        "train/episodic_length", info["episode"]["l"], self.global_step
                    )
                    # Episodic information from any one of the environments is sufficient 
                    break

            # Log steps per second (SPS) every self.log_freq steps
            if self.global_step % self.log_freq == 0:
                print("SPS:", int(self.global_step / (time.time() - start_time)))
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
        obs = self.envs.reset()

        while min(episode_count) < n_episodes:
            # Sample actions from the trained policy
            # actions, _, _ = self.sample_action(obs, to_numpy=True)
            actions = np.array(
                [self.envs.single_action_space.sample() for _ in range(self.n_envs)]
            )

            # Execute the actions in the environments
            next_obs, rewards, dones, _ = self.envs.step(actions)

            # Update the episode rewards
            for i, (reward, done) in enumerate(zip(rewards, dones)):
                episodic_returns[i] += reward

                if done and episode_count[i] < n_episodes:
                    print(
                        f"Environment {i+1}, Episode {episode_count[i] + 1}: Total Reward: {episodic_returns[i]}"
                    )
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
            obs = next_obs

        # Calculate and log the average returns over all test episodes for each environment
        for i, returns in enumerate(all_returns):
            avg_return = sum(returns) / len(returns)
            print(
                f"Environment {i+1}: Average Return over {n_episodes} episodes: {avg_return}"
            )
            self.writer.add_scalar(f"test/avg_return", avg_return, i + 1)

    def pad_source(self, x):
        """Padding on the starting ends of the series along sequence dimension
        with zero at the sequence boundaries to match source sequence length expected
        by representation model.

        Args:
            x (torch.FloatTensor): A source sequence.
                shape: (batch_size, seq_length, repr_dim)

        Returns:
            torch.FloatTensor: padded sequence if necessary else the original sequence.
        """
        # Calculate how much padding is needed
        padding_needed = self.src_seq_length - x.size(1)

        # Pad src sequence along the sequence dimension (dim=1) to match src_seq_length
        if padding_needed > 0:
            src = F.pad(src, (0, 0, padding_needed, 0), "constant", 0)

        return src


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

    envs = gym.make_env(env_id, seed)

    agent = CoreticAgent()
