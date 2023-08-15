import random
import time
from datetime import datetime

import gym
import numpy as np
import torch

from .agents import SACAgent
from .core.config import RLExperimentConfig, SACAgentConfig


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        if hasattr(env, "seed"):
            env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def set_torch_seed(seed, torch_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


if __name__ == "__main__":
    config = RLExperimentConfig()
    config.agent = SACAgentConfig()

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.name = f"{config.env_id}_{config.agent.name}_{current_datetime}"

    set_torch_seed(config.random_seed)

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                config.env_id, config.random_seed, 0, config.capture_video, config.name
            )
        ]
    )

    print("Number of environments: ", envs.num_envs)
    print("Observation Space: ", envs.single_observation_space)
    print("Action Space: ", envs.single_action_space)

    agent = SACAgent(
        envs=envs,
        critic_learning_rate=config.critic_optimizer.lr,
        actor_learning_rate=config.actor_optimizer.lr,
        buffer_size=config.replay_buffer.buffer_size,
        device=config.device,
    )

    start_time = time.time()

    config.total_timesteps = 10000
    agent.train(
        total_timesteps=config.total_timesteps,
        batch_size=config.replay_buffer.batch_size,
        learning_starts=config.learning_starts,
        alpha=config.agent.alpha,
        autotune=config.agent.autotune,
        gamma=config.agent.gamma,
        policy_frequency=config.agent.policy_frequency,
        target_network_frequency=config.agent.target_network_frequency,
        tau=config.agent.tau,
    )

    envs.close()
