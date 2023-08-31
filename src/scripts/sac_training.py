import time
from datetime import datetime

import gym
import numpy as np
import torch
from ..agents import SACAgent
from ..config import ReinforcedLearnerConfig, SACAgentConfig, OptimizerConfig
from ..envs.utils import make_env
from ..utils import set_torch_seed
from ..envs.normalization import RMVNormalizeVecObservation, EMANormalizeVecObservation
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    config = ReinforcedLearnerConfig(agent=SACAgentConfig(), total_timesteps=1e6)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.name = f"{config.env_id}_{config.agent.name}_{current_datetime}"

    set_torch_seed(config.random_seed)

    # Here only 1 environment
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id=config.env_id,
                seed=config.random_seed,
                idx=i,
                capture_video=config.capture_video,
                run_name=config.name,
            )
            for i in range(config.n_envs)
        ]
    )
    envs = RMVNormalizeVecObservation(
        envs, is_observation_scaling=config.normalize_observation
    )

    print("Number of environments: ", envs.num_envs)
    print("Observation Space: ", envs.single_observation_space)
    print("Action Space: ", envs.single_action_space)

    # Setup trial logging
    log_dir = config.checkpoint_dir / config.name
    writer = SummaryWriter(log_dir=log_dir)

    config_path = log_dir / "configuration.json"
    with open(config_path, "w") as config_file:
        config_file.write(config.model_dump_json(exclude={"device"}))

    agent = SACAgent(
        envs=envs,
        critic_learning_rate=config.agent.critic_optimizer.lr,
        actor_learning_rate=config.agent.actor_optimizer.lr,
        buffer_size=config.agent.buffer.buffer_size,
        device=config.device,
        writer=writer,
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
        tau=config.agent.tau,
    )

    agent.test(n_episodes=10)

    envs.close()
