import time
from datetime import datetime
from pathlib import PurePath

import carl
import numpy as np
import torch
from carl.envs import CARLPendulum
from carl.envs.dmc import CARLDmcQuadrupedEnv, CARLDmcWalkerEnv
from torch.utils.tensorboard import SummaryWriter

from ..agents import SACAgent
from ..config import OptimizerConfig, ReinforcedLearnerConfig, SACAgentConfig
from ..envs.core import make_env
from ..utils import get_action_dim, get_observation_dim, set_torch_seed

if __name__ == "__main__":
    config = ReinforcedLearnerConfig(
        # env_id='HalfCheetah-v2',
        env_id="CARL-test",
        batch_size=64,
        normalize_observation=True,
        agent=SACAgentConfig(),
        total_timesteps=1e6,
        device=torch.device("cuda:5" if torch.cuda.is_available() else "cpu"),
        n_envs=1
    )
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.name = f"{config.env_id}_{config.agent.name}_{current_datetime}"

    set_torch_seed(config.random_seed)

    # Here only 1 environment
    env = CARLDmcWalkerEnv(obs_context_as_dict=False)
    envs = make_env(
        # env=config.env_id,
        env=env,
        seed=config.random_seed,
        n_envs=config.n_envs,
        capture_video=config.capture_video,
        run_name=config.name,
        normalize_observation=config.normalize_observation,
    )

    print("Device in use: ", config.device)
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
    agent.save_checkpoint(PurePath(writer.get_logdir()))
    envs.close()
