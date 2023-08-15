import numpy as np
from .core import GenericAgent

class QLearningAgent(GenericAgent):
    def __init__(self, env, learning_rate, discount_factor, epsilon):
        super().__init__(env)
        self.q_table = np.zeros([self.envs.observation_space.n, self.envs.action_space.n])
        self.learning_rate = self.config['learning_rate']
        self.discount_factor = self.config['discount_factor']
        self.epsilon = self.config['epsilon']
    
    def sample_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return self.envs.action_space.sample()
        else:
            return np.argmax(self.q_table[state])
    
    def update_agent(self, experience):
        state, action, reward, next_state, done = experience
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
    
    def preprocess_experience(self, experience):
        # No preprocessing needed for a basic Q-learning agent
        return experience