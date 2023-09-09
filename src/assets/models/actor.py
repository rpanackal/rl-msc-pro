import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class Actor(nn.Module):
    def __init__(self, env, feat_dim):
        """Initialize the Actor model.

        Args:
            env: Gym environment. Used to get the observation and action dimensions.
        """
        super().__init__()
        action_dim = np.prod(env.single_action_space.shape)

        self.fc1 = nn.Linear(feat_dim, 256)
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
        mean = self.fc_mean(x)  # (-inf, inf)
        log_std = self.fc_logstd(x)  # (-inf, inf)
        log_std = self._squash_log_std(log_std) # (LOG_STD_MIN, LOG_STD_MAX)
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
        log_std = torch.tanh(log_std)  # (-1, 1)
        # log_std + 1 shift range to (0, 2)
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
        if torch.isnan(mean).any() or torch.isnan(std).any():
            print("NaN detected in mean or std")
        if torch.isnan(self.action_scale).any() or torch.isnan(self.action_bias).any():
            print("NaN detected in action_scale or action_bias")
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)  # Tanh squashing to bound the actions

        # Apply action scaling and bias
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforce action bound; this correction term is necessary when using tanh
        # squashing. This correcting the log_prob of x_t to log_prob y_t.
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)

        if torch.isnan(log_prob).any():
            print("NaN detected in log_prob")

        # you sum across all action dimensions to get a single scalar log_prob for 
        # the entire multi-dimensional action.
        log_prob = log_prob.sum(1, keepdim=True)
        squashed_mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, squashed_mean
