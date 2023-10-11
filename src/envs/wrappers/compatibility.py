import carl
import gym
import gymnasium as gymz
from shimmy.openai_gym_compatibility import _convert_space
from carl.envs.carl_env import CARLEnv

class EnvProjectCompatibility(gymz.Wrapper):
    """A wrapper that adds convenient methods and attributes to environments
    to work well with other environments and utilities.
    """
    def __init__(self, env: gymz.Env):
        
        self.is_contextual_env = True if isinstance(env, CARLEnv) else False
        env.metadata['is_contextual_env'] = self.is_contextual_env

        # CARLBraxEnv action space is gym.Space instead of gymz.Space, which is corrected 
        # for uniformity
        if hasattr(env,'action_space') and isinstance(env.action_space, gym.Space):
            env.action_space = _convert_space(env.action_space)
    
        super().__init__(env)
        

class VecEnvProjectCompatibility(gymz.vector.VectorEnvWrapper):
    """A wrapper that adds convenient methods and attributes to environments
    to work well with other environments and utilities.
    """
    def __init__(self, envs: gym.vector.VectorEnv):
        assert envs.is_vector_env, ValueError("The environment given is not vectorized.")
        self.is_contextual_env = envs.metadata.get('is_contextual_env', False)
        
        super().__init__(envs)
        # # CARLBraxEnv action space is gym.Space instead of gymz.Space, which is corrected 
        # # for uniformity
        # if hasattr(envs,'action_space') and isinstance(envs.action_space, gym.Space):
        #     envs.action_space = _convert_space(envs.action_space)