from datetime import datetime
from pathlib import PurePath

import torch
from shimmy.openai_gym_compatibility import _convert_space
from torch.utils.tensorboard import SummaryWriter

from ..envs.core import setup_env
from ..agents.adaptran import Adaptran
from ..agents.carl_sac import CARLSACAgent
from ..config import (
    BufferConfig,
    OptimizerConfig,
    ReinforcedLearnerConfig,
    SACAgentConfig,
)
from ..utils import get_action_dim, get_observation_dim, set_torch_seed

if __name__ == "__main__":
    config = ReinforcedLearnerConfig(
        env_id="CARLMountainCarContinuous",
        batch_size=512,
        normalize_observation=False,
        agent=SACAgentConfig(
            name="CARLSAC",
            actor_optimizer=OptimizerConfig(lr=3e-4),
            critic_optimizer=OptimizerConfig(lr=3e-4),
            policy_frequency=32,
            tau=0.01,
            autotune=False,
            alpha=0.1,
            gamma=0.9999,
            buffer=BufferConfig(buffer_size=5e4),
            expanse_dim=64,
        ),
        total_timesteps=1e5,
        device=torch.device("cuda:6" if torch.cuda.is_available() else "cpu"),
        n_envs=1,
        learning_starts=1e3,
        random_seed=60,
    )
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.name = f"{config.env_id}_{config.agent.name}_{current_datetime}"

    set_torch_seed(config.random_seed)

    envs = setup_env(
        env_cls_name=config.env_id,
        n_envs=config.n_envs,
        seed=config.random_seed,
        # env_suite=args.package,
        normalize_observation=config.normalize_observation,
    )

    print("Device in use: ", config.device)
    print("Number of environments: ", envs.num_envs)
    print("Observation Space: ", envs.observation_space)
    print("Action Space: ", envs.observation_space)

    # Setup trial logging
    log_dir = config.checkpoint_dir / config.name
    writer = SummaryWriter(log_dir=log_dir)

    config_path = log_dir / "configuration.json"
    with open(config_path, "w") as config_file:
        config_file.write(config.model_dump_json(exclude={"device"}))

    agent = CARLSACAgent(
        envs=envs,
        critic_learning_rate=config.agent.critic_optimizer.lr,
        actor_learning_rate=config.agent.actor_optimizer.lr,
        buffer_size=config.agent.buffer.buffer_size,
        device=config.device,
        writer=writer,
        log_freq=config.agent.log_freq,
        expanse_dim=config.agent.expanse_dim,
        seed=config.random_seed,
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

    # agent.test(n_episodes=10)
    # agent.save_checkpoint(PurePath(writer.get_logdir()))
    agent.replay_buffer.save(PurePath(writer.get_logdir()))
    envs.close()
