import gym
import numpy as np
from abc import ABC, abstractmethod

class GenericAgent(ABC):
    def __init__(self, env):
        self.envs = env

    @abstractmethod
    def sample_action(self, state):
        """Sample action from the policy given current state."""
        pass

    @abstractmethod
    def update_agent(self, experience):
        """Update the agent (e.g., Q-network, policy, value function) based on experience."""
        pass

    @abstractmethod
    def preprocess_experience(self, experience):
        """Preprocess experience if needed (e.g., stacking frames, normalizing)."""
        return experience

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.envs.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.sample_action(state)
                next_state, reward, done, _ = self.envs.step(action)
                experience = (state, action, reward, next_state, done)
                experience = self.preprocess_experience(experience)
                self.update_agent(experience)

                episode_reward += reward
                state = next_state
            
            print(f"Episode {episode}: Reward = {episode_reward}")

    def test(self, num_episodes):
        for episode in range(num_episodes):
            state = self.envs.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.sample_action(state)
                next_state, reward, done, _ = self.envs.step(action)
                episode_reward += reward
                state = next_state
            
            print(f"Episode {episode}: Reward = {episode_reward}")
