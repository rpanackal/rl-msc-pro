import time

import gym
import numpy as np
import torch
import torch.nn.functional as F
from gym.vector import VectorEnv
from ..data.buffer import EpisodicBuffer, EpisodicBufferSamples
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from ..assets.models import Actor, SoftQNetwork
from .core import GenericAgent, CompactStateTransitions


class SACwReprModel(GenericAgent):
    """
    An simple SAC agent coupled with represetnation learning model.
    """

    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv | gym.Env,
        repr_model,
        repr_model_learning_rate: float,
        critic_learning_rate: float,
        actor_learning_rate: float,
        buffer_size: int,
        device: torch.device,
        writer: SummaryWriter | None = None,
        log_freq: int = 100,
    ):
        assert isinstance(
            envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"

        if hasattr(envs, "num_envs"):
            envs.single_observation_space.dtype = np.float32
            self.observation_dim = np.prod(envs.single_observation_space.shape)
            self.action_dim = np.prod(envs.single_action_space.shape)
            self.n_envs = envs.num_envs
        else:
            envs.observation_space.dtype = np.float32
            self.observation_dim = np.prod(envs.observation_space.shape)
            self.action_dim = np.prod(envs.action_space.shape)
            self.n_envs = 1

        self.repr_model = repr_model
        self.repr_model.eval()

        self.src_seq_length = self.repr_model.src_seq_length
        self.tgt_seq_length = self.repr_model.tgt_seq_length
        self.embed_dim = self.repr_model.embed_dim
        self.repr_dim = self.embed_dim * self.src_seq_length

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

        self.replay_buffer = EpisodicBuffer(
            buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            n_envs=self.n_envs,
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
        kappa: float = 0.01,
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
            kappa (float): The probability under which padded source sequences are used for training
                generating states. In the first src_seq_len steps of an episode, zero padding is needed
                as complete source sequence doesn't exit. This hyperparameter controls the probability
                of paded source sequences are used for generating states, as it might be crucial to
                generate robust state representation at episode starts.
        """

        self.autotune = autotune

        assert not (alpha is None and not autotune), ValueError(
            "If alpha is not given, autotune has to be enabled."
        )

        # If automatic tuning of alpha is enabled, set up the corresponding variables and
        # optimizers
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
        self.kappa = kappa

        # Initialize the global step counter
        self.global_step = 0

    def sample_action(self, states):
        """
        Sample an action from the policy given the current observations.

        Args:
            states: The current state states from the environment.
                shape: (batch_size, state_seq_len, embed_dim)
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
            assert states is not None, ValueError(
                "Actor doesn't accept None valued states"
            )
            # Convert observations to Tensor if they're not already
            if not isinstance(states, torch.Tensor):
                states = torch.Tensor(states).to(self.device)

            # Use the actor model to sample actions, log_probs, and squashed_means
            actions, log_probs, squashed_means = self.actor.get_action(states)
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
        batch: EpisodicBufferSamples = self.replay_buffer.sample(
            self.batch_size, desired_length=self.src_seq_length + 1
        )
        state_transitions = self.get_states_transitions(batch)

        # Update the Critic network
        self.update_critic(state_transitions)

        # Update the Actor network
        # Only update the actor network periodically, as defined by policy_frequency
        if self.global_step % self.policy_frequency == 0:
            self.update_actor(state_transitions)

        # Update the Target Networks
        # Only update the target networks periodically, as defined by target_network_frequency
        if self.global_step % self.target_network_frequency == 0:
            self.update_target_networks()

    def to_torch(self, x):
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x).to(self.device)
        return x

    def get_sources(self, observations, actions, dim=-1):
        return torch.cat([observations, actions], dim=dim)

    def shift_right_fill(self, x: torch.Tensor, shift=1, dim=1, fill=0):
        # Get the shape of the original tensor
        n_dim = x.ndim

        # Update the shape to reflect the padding on the given dimension and shift
        padding_shape = [0] * n_dim * 2
        padding_shape[dim * 2] = shift

        # Pad with zeros and slice to achieve the "shifting" effect
        x = F.pad(x, padding_shape, "constant", fill)
        slicing_cmd = [slice(None, -shift if d == dim else None) for d in range(n_dim)]
        x = x[slicing_cmd]
        return x

    def get_states_transitions(self, batch: EpisodicBufferSamples, kappa=0.01):
        state_seq_length = batch.observations.size(1) - self.src_seq_length + 1
        # 1. Select the index (inclusive) from which sources are consucutively
        # sampled.

        # For a probability kappa, sample a random time step
        # for first source
        t = (
            torch.randint(high=self.src_seq_length, size=(1,)).item()
            if torch.rand(1).item() <= kappa
            else self.src_seq_length - 1
        )
        state_transitions = CompactStateTransitions(
            states=torch.empty((self.batch_size, state_seq_length, self.repr_dim)).to(
                self.device
            ),
            actions=batch.actions[:, t : t + state_seq_length, :],
            rewards=batch.rewards[:, t : t + state_seq_length],
            dones=batch.dones[:, t : t + state_seq_length],
        )

        sources = self.get_sources(batch.observations, batch.actions)

        shift = self.src_seq_length - t + 1
        if shift:
            sources = self.shift_right_fill(sources, shift)

        assert sources.size(1) == batch.observations.size(1), ValueError(
            "The source creation logic is flawed"
        )

        for tau in range(self.src_seq_length, self.src_seq_length + state_seq_length):
            source = sources[:, tau - self.src_seq_length : tau, :].clone()
            source[:, -1, -self.action_dim :] = 0

            enc_output, mean, logvar, latent = self.repr_model(source, enc_only=True)

            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            self.repr_optimizer.zero_grad()
            kl_loss.backward()
            self.repr_optimizer.step()

            with torch.no_grad():
                state_transitions.states[
                    :, tau - self.src_seq_length, :
                ] = torch.flatten(enc_output, start_dim=1, end_dim=2)

        if self.global_step % self.log_freq == 0 and self.writer:
            self.writer.add_scalar("losses/repr_loss", kl_loss.item(), self.global_step)

        return state_transitions

    def update_critic(self, state_transitions: CompactStateTransitions):
        """
        Updates the critic network using the given state transitions.

        Args:
            state_transitions (CompactStateTransitions): A sequence of continuous transitions
                of compact state, action, reward and dones each of shape (batch_size,
                state_seq_len, ...).
        """
        state_seq_length = state_transitions.states.size(1)

        for t in range(state_seq_length - 1):
            with torch.no_grad():
                # 1. Sample next action and compute Q-value targets
                next_state_actions, next_state_log_pi, _ = self.sample_action(
                    state_transitions.states[:, t + 1, :]
                )
                qf1_next_target = self.qf1_target(
                    state_transitions.states[:, t + 1, :], next_state_actions
                )
                qf2_next_target = self.qf2_target(
                    state_transitions.states[:, t + 1, :], next_state_actions
                )

                # 2. Compute the minimum Q-value target and the target for the Q-function
                # update
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - self.alpha * next_state_log_pi
                )
                next_q_value = state_transitions.rewards[:, t].flatten() + (
                    1 - state_transitions.dones[:, t].flatten()
                ) * self.gamma * min_qf_next_target.view(-1)

            # 3. Compute the Q-values and the MSE loss for both critics
            qf1_a_values = self.qf1(
                state_transitions.states[:, t, :], state_transitions.actions[:, t, :]
            ).view(-1)
            qf2_a_values = self.qf2(
                state_transitions.states[:, t, :], state_transitions.actions[:, t, :]
            ).view(-1)
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
                self.writer.add_scalar(
                    "losses/qf1_loss", qf1_loss.item(), self.global_step
                )
                self.writer.add_scalar(
                    "losses/qf2_loss", qf2_loss.item(), self.global_step
                )
                self.writer.add_scalar(
                    "losses/qf_loss", qf_loss.item() / 2.0, self.global_step
                )

    def update_actor(self, state_transitions: CompactStateTransitions):
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
        state_seq_length = state_transitions.states.size(1)

        # The loop here runs multiple updates for each actor update,
        # which is specified by self.policy_frequency.
        for _ in range(self.policy_frequency):
            t = torch.randint(high=state_seq_length, size=(1,)).item()
            # 1. Sample Actions:
            # Using the current policy, sample actions and their log
            # probabilities from the current state batch.
            pi, log_pi, _ = self.sample_action(state_transitions.states[:, t, :])

            # 2. Compute the Q-values of the sampled actions using both the
            # Q-functions (Q1 and Q2).
            qf1_pi = self.qf1(state_transitions.states[:, t, :], pi)
            qf2_pi = self.qf2(state_transitions.states[:, t, :], pi)

            # Take the minimum Q-value among the two Q-functions to improve
            # robustness.
            min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)

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
                    _, log_pi, _ = self.sample_action(state_transitions.states[:, t, :])

                # Compute the loss for alpha, aiming to keep policy entropy
                # close to target_entropy.
                alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()

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
        kappa: float = 0.01,
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
            kappa (float): The probability under which padded source sequences are used for training
                generating states. In the first src_seq_len steps of an episode, zero padding is needed
                as complete source sequence doesn't exit. This hyperparameter controls the probability
                of paded source sequences are used for generating states, as it might be crucial to
                generate robust state representation at episode starts.

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
            kappa=kappa,
        )

        # Start timer
        start_time = time.time()

        # Reset the environment and get initial observation
        obs = self.envs.reset()
        for _ in range(total_timesteps):
            # Sample actions
            with torch.no_grad():
                states = self.get_current_state(obs)
                actions, _, _ = self.sample_action(states)

            actions = (
                actions.cpu().numpy()
                if not isinstance(actions, np.ndarray)
                else actions
            )

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
                    if self.writer:
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
                    # Episodic information from any one of the environments is sufficient
                    break

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

    def get_current_state(self, curr_obs: np.ndarray):
        # If learning hasn't started, current states are unnecessary
        # as actions are randomly sampled.
        if self.global_step < self.learning_starts:
            return

        # Fetch the ongoing episode form buffer
        batch: EpisodicBufferSamples = self.replay_buffer.get_last_episode()

        source = torch.zeros(
            (
                self.n_envs,
                self.src_seq_length,
                self.observation_dim + self.action_dim,
            )
        )

        # Create source sequences for each environment
        for env_no in range(self.n_envs):
            ep_len = batch.observations[env_no].shape[0]

            # We need 1 less than the src_seq_length as curr_obs is included
            # later
            start = ep_len - self.src_seq_length + 1
            start = start if start >= 0 else 0

            past_obs, past_acts = (
                batch.observations[env_no][start:],
                batch.actions[env_no][start:],
            )
            past_length = len(past_obs)

            if past_length > 0:
                source[env_no, -(past_length + 1) :] = torch.tensor(
                    np.block(
                        [
                            [past_obs, past_acts],
                            [curr_obs[env_no], np.zeros(self.action_dim)],
                        ]
                    )
                )
            else:
                source[env_no, -1:] = torch.tensor(
                    np.block([[curr_obs[env_no], np.zeros(self.action_dim)]])
                )

        source = source.to(self.device).float()

        # Get state representations
        enc_output, mean, logvar, latent = self.repr_model(source, enc_only=True)
        state = torch.flatten(enc_output, start_dim=1, end_dim=2)

        return state

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
            actions, _, _ = self.sample_action(obs, to_numpy=True)

            # Execute the actions in the environments
            next_obs, rewards, dones, _ = self.envs.step(actions)

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
            obs = next_obs

        # Calculate and log the average returns over all test episodes for each environment
        for i, returns in enumerate(all_returns):
            avg_return = sum(returns) / len(returns)
            print(
                f"Environment {i+1}: Average Return over {n_episodes} episodes: {avg_return}"
            )
            if self.writer:
                self.writer.add_scalar(f"test/avg_return", avg_return, i + 1)


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

    agent = SACAgent()
