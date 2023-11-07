import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class SoftQNetwork(nn.Module):
    def __init__(self, env, feat_dim, expanse_dim=256):
        """Initialize the Soft Q-Network.

        Args:
            env: Gym environment. Used to get the observation and action dimensions.
        """
        super().__init__()
        self.env = env
        self.observation_dim = feat_dim

        if hasattr(env, 'single_action_space'):
            space = env.single_action_space
        else:
            space = env.action_space

        # Calculate total input size from observation and action space
        input_dim = feat_dim + np.prod(
            space.shape
        )

        # Define the neural network layers
        self.fc1 = nn.Linear(input_dim, expanse_dim)
        self.fc2 = nn.Linear(expanse_dim, expanse_dim)
        self.fc3 = nn.Linear(expanse_dim, 1)

    def forward(self, x, a):
        """Pass the observation and action through the neural network to get the Q-value.

        Args:
            x: Observation tensor.
            a: Action tensor.

        Returns:
            Q-value for the given observation and action.
        """
        # Concatenate the observation and action to form the input
        x = torch.cat([x, a], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def model_twin(self):
        """Return a new instance of the same class with the same configuration."""
        return self.__class__(self.env, self.observation_dim)
