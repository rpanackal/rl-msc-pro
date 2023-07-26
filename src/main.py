import gym
#import gymnasium as gym
import d4rl # Import required to register environments, you may need to also import the submodule
from d4rl import gym_mujoco
import time
import collections
from .utils import inspect_head_dataset, sequence_dataset
import numpy as np


def get_sequence_dataset(env, seq_length=None):
    key_features = ("observations", "actions", "rewards")
    
    prev_episode_length = None
    dataset = []

    for episode in sequence_dataset(env):
        episode_length = len(episode["rewards"])

        # When seq_length is None, episode lengths are preserved
        seq_length = seq_length or episode_length

        assert episode_length % seq_length == 0, ValueError(f"Episode length \
            ({episode_length}) are not divisible by sequence length ({seq_length})")
        
        n_parts = episode_length // seq_length
        # Split observations, actions and rewards along time dimension 
        # (episode_length, feat_dim) -> (n_parts, seq_length, feat_dim)
        parts = [episode[key].reshape(n_parts, seq_length, -1) for key in key_features]

        # Split observations, actions and rewards along feature dimension 
        dataset.append(np.concatenate(parts, axis=2))

        if prev_episode_length and prev_episode_length != episode_length:
            raise ValueError("Episodes are of different lengths.")
        else:
            prev_episode_length = episode_length

    return np.concatenate(dataset, axis=0)


def main():
    # Create the environment
    env = gym.make('halfcheetah-medium-v2')
    
    obs_space = env.observation_space
    action_space = env.action_space

    print(f"The observation space: env - {obs_space}")
    print(f"The action space: env - {action_space}")
    
    print(f"Maximum episode space: env - {env._max_episode_steps}")

    
    dataset = get_sequence_dataset(env)
    print(dataset.shape)

if __name__ == "__main__":
    main()
    