import time

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.vector import VectorEnv
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from ..assets import Transformer
from ..assets.models import Actor, SoftQNetwork
from ..data.buffer import EpisodicBuffer, EpisodicBufferSamples
from .core import GenericAgent, CompactStateTransitions
from ..utils import get_observation_dim


class CoretranAgent(GenericAgent):
    """Contextual Representation Learning via Time Series Transformers for Control" """

    def __init__(
        self,
        envs: gymnasium.vector.SyncVectorEnv | gymnasium.Env,
        repr_model: Transformer,
        repr_model_learning_rate: float,
        critic_learning_rate: float,
        actor_learning_rate: float,
        buffer_size: int,
        device: torch.device,
        writer: SummaryWriter | None = None,
        log_freq: int = 100,
    ):
        assert isinstance(envs.single_action_space, gymnasium.spaces.Box), ValueError(
            f"only continuous action space is supported, given {envs.single_action_space}"
        )
        if getattr(envs, "is_vector_env", None):
            if envs.is_contextual_env:
                envs.single_observation_space["obs"].dtype = np.float32
                envs.single_observation_space["context"].dtype = np.float32
            else:
                envs.single_observation_space.dtype = np.float32
            self.observation_dim = get_observation_dim(envs)
            self.action_dim = np.prod(envs.single_action_space.shape)
            self.n_envs = envs.num_envs
        else:
            if envs.is_contextual_env:
                envs.observation_space["obs"].dtype = np.float32
                envs.observation_space["context"].dtype = np.float32
            else:
                envs.observation_space.dtype = np.float32
            self.observation_dim = get_observation_dim(envs)
            self.action_dim = np.prod(envs.action_space.shape)
            self.n_envs = 1

        self.repr_model = repr_model.float()
        self.repr_model_target = repr_model.model_twin().to(device)
        self.repr_model_target.load_state_dict(self.repr_model.state_dict())
        self.repr_model_target.eval()

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
            buffer_size=buffer_size,
            observation_space=envs.single_observation_space,
            action_space=envs.single_action_space,
            device=device,
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
        state_seq_length: int = 2,
        kappa: float = 0.01,
        kl_weight: float = 0.5,
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
            state_seq_length (int): The length of state sequences produced by representation
                model for critic update. Directly impact training time.
            kappa (float): The probability under which padded source sequences are used for training
                generating states. In the first src_seq_len steps of an episode, zero padding is needed
                as complete source sequence doesn't exit. This hyperparameter controls the probability
                of paded source sequences are used for generating states, as it might be crucial to
                generate robust state representation at episode starts.

                Heuristic: (source sequence length * 10) / average episode
            kl_weight (float): A scalar term weight of KL divergence term against the reconstruction
                loss term in the overall representation model loss function.
        Raises:
            ValueError: If `state_seq_length` is too small for critic update. Minimum allowed
                value is 2.
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
        self.kappa = kappa
        self.kl_weight = kl_weight

        assert state_seq_length >= 2, ValueError(
            f"State sequence length ({state_seq_length}) too small for critic update. \
            Minimum allowed value is 2"
        )
        self.state_seq_length = state_seq_length

        # Minimum length of trajectory sampled from replay buffer
        self.min_trajectory_length = self.src_seq_length + self.tgt_seq_length

        # Initialize the global step counter
        self.global_step = 0

    def sample_action(self, states):
        """
        Sample an action from the policy given the current states.

        Args:
            states: The current state states from the environment.
                shape: (batch_size, state_seq_len, repr_dim)
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
            # Convert states to Tensor if they're not already
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
        # Create a copy of next_obs to avoid modifying the original one
        real_next_obs = next_obs.copy()

        # Loop over each sub-environment using the 'dones' array
        for idx, done in enumerate(dones):
            if done:  # if the sub-environment has terminated
                if self.envs.is_contextual_env:
                    real_next_obs[idx] = infos["final_observation"][idx]["obs"]
                else:
                    real_next_obs[idx] = infos["final_observation"][idx]

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
        desired_length = self.min_trajectory_length + self.state_seq_length - 1
        samples = self.replay_buffer.sample(self.batch_size, desired_length)

        # Update Represenation Model
        state_transitions = self.update_repr_model(samples)

        # Update the Critic network
        self.update_critic(state_transitions)

        # Update the Actor network
        # Only update the actor network periodically, as defined by policy_frequency
        if self.global_step % self.policy_frequency == 0:
            # Sample a batch of random episodes/trajectories from episodic replay buffer
            # samples = self.replay_buffer.sample(
            #     self.batch_size, self.min_trajectory_length
            # )
            # self.update_actor(samples, state_transitions)

            self.update_actor(state_transitions)

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
            state_transitions (CompactStateTransitions): A sequence of continuous transitions
                of compact state, action, reward and dones each of shape (batch_size,
                state_seq_len, ...).
        """
        # 1. Compute all possible source sequences in trajectory for which
        # there exist a target sequence in the sampled trajectory.

        # Padd zeros to the left of sequences.
        zeroes = torch.zeros(
            (
                self.batch_size,
                self.src_seq_length - 1,
                self.observation_dim + self.action_dim,
            )
        ).to(self.device)
        sources = torch.cat(
            [
                zeroes,
                # Concatenate actions and observations along feat_dim
                torch.cat(
                    [
                        samples.observations[
                            :, : self.src_seq_length + self.state_seq_length - 1, :
                        ],
                        samples.actions[
                            :, : self.src_seq_length + self.state_seq_length - 1, :
                        ],
                    ],
                    dim=2,
                ),
            ],
            dim=1,
        )

        # 2. Compute target all target sequences in trajectory
        # targets = samples.observations[:, 1:, :]
        # conditionals = samples.actions[:, 1:, :]

        # 3. Compute index in sources to start sampling source from.
        start_atmost = self.src_seq_length - 1
        start_atleast = 0
        # With a probability  1 - eta, start is start_atmost, else,
        # start is uniformly from anywhere in 0..start_atmost
        start = (
            torch.randint(start_atleast, start_atmost + 1, (1,)).item()
            if torch.rand(1).item() < self.kappa
            else (start_atmost)
        )
        end = start + self.state_seq_length

        state_transitions = CompactStateTransitions(
            states=torch.empty(
                (self.batch_size, self.state_seq_length, self.repr_dim)
            ).to(self.device),
            next_states=torch.empty(
                (self.batch_size, self.state_seq_length - 1, self.repr_dim)
            ).to(self.device),
            actions=samples.actions[:, start:end, :],
            rewards=samples.rewards[:, start:end],
            dones=samples.dones[:, start:end],
            loss=None,
        )

        self.repr_model.train()
        # Loop over state_seq_len source sequences from start.
        for t in range(start, end):
            # Get source, target and conditional for current time step
            source = sources[:, t : t + self.src_seq_length, :].clone()
            # During online interaction, actions are sampled and not available for state calculation
            # for corresponding time step
            source[:, -1, -self.action_dim :] = 0

            # target = targets[:, t : t + self.tgt_seq_length, :]
            # conditional = conditionals[:, t : t + self.tgt_seq_length, :]

            # 3. Produce state representations and future observations
            dec_output, enc_output = self.repr_model(source, full_output=True)

            # 5. Compute representation model loss
            mse_loss = F.mse_loss(dec_output, source)
            # kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            total_loss = mse_loss  # + self.kl_weight * kl_loss

            # # 6. Perform backpropagation to update the representation learning model.
            # self.repr_optimizer.zero_grad()
            # total_loss.backward()
            # self.repr_optimizer.step()

            # Log Q-function loss information of last state in sequence every log_freq steps.
            if self.global_step % self.log_freq == 0 and self.writer:
                self.writer.add_scalar(
                    "losses/repr_loss", total_loss.item(), self.global_step
                )

            state_transitions.states[:, t - start, :] = torch.flatten(
                enc_output, start_dim=1, end_dim=2
            )
            # state_transitions.states[:, t - start, :] = enc_output[:, -1, :]
            # ? Should loss be added across all state_seq_length time steps ?
            state_transitions.loss = total_loss
            # 4. Store final element along sequence dimension of enc_output as the state s_t,
            # Map times steps from source to time steps in trajectory.
            with torch.no_grad():
                if t + 1 < end:
                    source = sources[:, t + 1 : t + 1 + self.src_seq_length, :].clone()
                    # During online interaction, actions are sampled and not available for state calculation
                    # for corresponding time step
                    source[:, -1, -self.action_dim :] = 0
                    enc_output = self.repr_model_target(source, enc_only=True)
                    state_transitions.next_states[:, t - start, :] = torch.flatten(
                        enc_output, start_dim=1, end_dim=2
                    )
                    # state_transitions.next_states[:, t - start, :] = enc_output[
                    #     :, -1, :
                    # ]
        # ? Should padding be done on target to get state sequence for all observations?
        return state_transitions

    def update_critic(self, state_transitions: CompactStateTransitions):
        """
        Updates the critic network using the given state transitions.

        Args:
            state_transitions (CompactStateTransitions): A sequence of continuous transitions
                of compact state, action, reward and dones each of shape (batch_size,
                state_seq_len, ...).
        """
        # Initialize loss and current states

        # Loop through each time step in the trajectory
        # Note: We don't have next_states for the last time step in each trajectory,
        # so we loop until the (n_src_per_batch - 1)th time step.
        # ? Should loop be replaced with reshape to perform all operations at once?
        for t in range(self.state_seq_length - 1):
            with torch.no_grad():
                # 2. Sample next action and compute Q-value targets
                next_state_action, next_state_log_pi, _ = self.sample_action(
                    state_transitions.next_states[:, t, :]
                )
                qf1_next_target = self.qf1_target(
                    state_transitions.next_states[:, t, :], next_state_action
                )
                qf2_next_target = self.qf2_target(
                    state_transitions.next_states[:, t, :], next_state_action
                )

                # 3. Compute the minimum Q-value target and the target for the Q-function update
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - self.alpha * next_state_log_pi
                )
                next_q_value = state_transitions.rewards[:, t].flatten() + (
                    1 - state_transitions.dones[:, t].flatten()
                ) * self.gamma * min_qf_next_target.view(-1)

            # 4. Compute the Q-values and the MSE loss for both critics
            qf1_a_value = self.qf1(
                state_transitions.states[:, t, :], state_transitions.actions[:, t, :]
            ).view(-1)
            qf2_a_value = self.qf2(
                state_transitions.states[:, t, :], state_transitions.actions[:, t, :]
            ).view(-1)
            qf1_loss = F.mse_loss(qf1_a_value, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_value, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            total_loss = qf_loss + state_transitions.loss

            # 5. Update critic model
            self.repr_optimizer.zero_grad()
            self.q_optimizer.zero_grad()
            # qf_loss.backward()
            total_loss.backward()
            self.q_optimizer.step()
            self.repr_optimizer.step()

        # Log Q-function loss information of last state in sequence every log_freq steps.
        if self.global_step % self.log_freq == 0 and self.writer:
            self.writer.add_scalar(
                "losses/qf1_values", qf1_a_value.mean().item(), self.global_step
            )
            self.writer.add_scalar(
                "losses/qf2_values", qf2_a_value.mean().item(), self.global_step
            )
            self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), self.global_step)
            self.writer.add_scalar("losses/qf2_loss", qf2_loss.item(), self.global_step)
            self.writer.add_scalar(
                "losses/qf_loss", qf_loss.item() / 2.0, self.global_step
            )

    def update_actor(self, state_transitions: CompactStateTransitions):
        """
        Updates the actor network using the given state transitions.

        Args:
            state_transitions (CompactStateTransitions): A sequence of continuous transitions
                of compact state, action, reward and dones each of shape (batch_size,
                state_seq_len, ...).

        Returns:
            None: The method updates the actor network and optionally the temperature parameter
                in-place.
        """

        # 4. Store final element along sequence dimension of enc_output as the state s_t,
        t = torch.randint(high=self.state_seq_length, size=(1,)).item()
        state = state_transitions.states[:, t, :].detach()

        # START: Policy freq loop
        for _ in range(self.policy_frequency):
            # 4. Sample Actions:
            # Using the current policy, sample actions and their log
            # probabilities from the current batch of states.
            pi, log_pi, _ = self.sample_action(state)

            # 5. Compute the Q-values of the sampled actions using both the
            # Q-functions (Q1 and Q2).
            qf1_pi = self.qf1(state, pi)
            qf2_pi = self.qf2(state, pi)

            # Take the minimum Q-value among the two Q-functions to improve
            # robustness.
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            # 6. Compute Actor Loss:
            # The actor aims to maximize this quantity, which corresponds
            # to maximizing Q-value and entropy.
            actor_loss = (self.alpha * log_pi - min_qf_pi).mean()

            # 7. Perform backpropagation to update the actor network.
            # self.repr_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # self.repr_optimizer.step()

            # 8. Update Temperature (Optional):
            # If self.autotune is True, the temperature parameter alpha is
            # also learned.
            if self.autotune:
                # Sample actions again (not strictly needed, could reuse above)
                with torch.no_grad():
                    _, log_pi, _ = self.sample_action(state)

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
        # END: Policy freq loop

        # Log Actor loss and alpha of last state in sequence every self.log_freq steps.
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
        # Q-function targets
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

        # Representation model targets
        for param, target_param in zip(
            self.repr_model.parameters(), self.repr_model_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def get_current_state(self, curr_obs: np.ndarray):
        # If learning hasn't started, current states are unnecessary
        # as actions are randomly sampled.
        if self.global_step < self.learning_starts:
            return

        # Fetch the ongoing episode form buffer
        samples = self.replay_buffer.get_last_episode()

        source = torch.zeros(
            (
                self.n_envs,
                self.src_seq_length,
                self.observation_dim + self.action_dim,
            )
        )

        # Create source sequences for each environment
        for env_no in range(self.n_envs):
            ep_len = samples.observations[env_no].shape[0]

            start = ep_len - self.src_seq_length + 1
            start = start if start >= 0 else 0

            past_obs, past_acts = (
                samples.observations[env_no][start:],
                samples.actions[env_no][start:],
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
                source[env_no, -(past_length + 1) :] = torch.tensor(
                    np.block([[curr_obs[env_no], np.zeros(self.action_dim)]])
                )

        source = source.to(self.device).float()

        # Get state representations
        # self.repr_model.eval()
        with torch.no_grad():
            enc_output = self.repr_model_target(source, enc_only=True)

        state = torch.flatten(enc_output, start_dim=1, end_dim=2)
        # state = enc_output[:, -1, :]
        return state

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
        state_seq_length: int = 2,
        kappa: float = 0.01,
        kl_weight: float = 0.5,
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
            state_seq_length (int | None): The length of state sequences produced by representation
                model for critic update. Directly impact training time.
            kappa (float): The probability under which padded source sequences are used for training
                generating states. In the first src_seq_len steps of an episode, zero padding is needed
                as complete source sequence doesn't exit. This hyperparameter controls the probability
                of paded source sequences are used for generating states, as it might be crucial to
                generate robust state representation at episode starts.

                Heuristic: (source sequence length * 10) / average episode length
            kl_weight (float): A scalar term weight of KL divergence term against the reconstruction
                loss term in the overall representation model loss function.
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
            state_seq_length=state_seq_length,
            kappa=kappa,
            kl_weight=kl_weight,
        )

        # Start timer
        start_time = time.time()

        # Reset the environment and get initial observation
        curr_obs, _ = self.envs.reset()
        if self.envs.is_contextual_env:
            curr_obs, context = curr_obs["obs"], curr_obs["context"]

        for _ in range(total_timesteps):
            # Sample actions
            with torch.no_grad():
                states = self.get_current_state(curr_obs)
                actions, _, _ = self.sample_action(states)

            actions = (
                actions.cpu().numpy() if not isinstance(actions, np.ndarray) else actions
            )
            # Execute actions in the environment
            next_obs, rewards, dones, infos = self.envs.step(actions)
            if self.envs.is_contextual_env:
                next_obs, context = next_obs["obs"], next_obs["context"]

            experience = (curr_obs, next_obs, actions, rewards, dones, infos)

            # Prepare experience for agent update
            experience = self.preprocess_experience(experience)

            # Update the agent
            self.update_agent(experience)

            # Update the current observation
            curr_obs = next_obs

            # Log episodic information if available
            # Episodic information from any one of the environments is sufficient
            if "episode" in infos:
                print(
                    f"global_step={self.global_step}, episodic_return={infos['episode']['r'][0]}"
                )
                if self.writer:
                    self.writer.add_scalar(
                        "train/episodic_return",
                        infos["episode"]["r"][0],
                        self.global_step,
                    )
                    self.writer.add_scalar(
                        "train/episodic_length",
                        infos["episode"]["l"][0],
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
        # TODO: Update for sequential experience
        # Initialize a list to keep track of all returns for each environment
        # and number of episodes progresed in each of them.
        all_returns = [[] for _ in range(self.n_envs)]
        episode_count = [0] * self.n_envs

        # Initialize a list to keep track of returns for ongoing episode in each environment
        episodic_returns = [0] * self.envs.num_envs

        # Reset the environments to get an initial observation state
        obs, _ = self.envs.reset()

        while min(episode_count) < n_episodes:
            # Sample actions from the trained policy
            with torch.no_grad():
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

    envs = gymnasium.make_env(env_id, seed)

    agent = CoretranAgent()
