import gym
import d4rl # Import required to register environments, you may need to also import the submodule

# Create the environment
env = gym.make('ant-random-v2')

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()

print(dataset.keys())
#print(dataset['observations']) # An N x dim_observation Numpy array of observations
