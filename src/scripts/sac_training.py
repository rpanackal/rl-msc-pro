import time
from datetime import datetime

import gym
import numpy as np
import torch

from ..agents import SACAgent
from ..config import ReinforcedLearnerConfig, SACAgentConfig
from ..envs.utils import make_env
from ..utils import set_torch_seed

from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    config = ReinforcedLearnerConfig(
        agent=SACAgentConfig(),
        total_timesteps=10000
    )
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.name = f"{config.env_id}_{config.agent.name}_{current_datetime}"

    set_torch_seed(config.random_seed)

    # Here only 1 environment as list contains only one function
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

    # Setup trial logging
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    trial_name = f"{config.name}_{config.agent.name}_{current_datetime}"
    log_dir = config.checkpoint_dir / trial_name
    writer = SummaryWriter(log_dir=log_dir)

    agent = SACAgent(
        envs=envs,
        critic_learning_rate=config.agent.critic_optimizer.lr,
        actor_learning_rate=config.agent.actor_optimizer.lr,
        buffer_size=config.agent.buffer.buffer_size,
        device=config.device,
        writer=writer
    )

    
    agent.train(
        total_timesteps=config.total_timesteps,
        batch_size=config.batch_size,
        learning_starts=config.learning_starts,
        alpha=config.agent.alpha,
        autotune=config.agent.autotune,
        gamma=config.agent.gamma,
        policy_frequency=config.agent.policy_frequency,
        target_network_frequency=config.agent.target_network_frequency,
        tau=config.agent.tau
    )

    agent.test(n_episodes=10)
    
    envs.close()
