import time
from datetime import datetime
from pathlib import PurePath
from typing import Type, Union

import carl
import gym
import gymnasium as gymz
import numpy as np
import torch
from carl.envs import (CARLBraxHalfcheetah, CARLDmcQuadrupedEnv,
                       CARLDmcWalkerEnv, CARLMountainCarContinuous,
                       CARLPendulum)
from carl.envs.dmc.carl_dmcontrol import CARLDmcEnv
from gymnasium.wrappers import (FilterObservation, FlattenObservation,
                                RecordEpisodeStatistics)
from shimmy.openai_gym_compatibility import _convert_space
from torch.utils.tensorboard import SummaryWriter

from ..agents.carl_sac import CARLSACAgent
from ..agents.adaptran_v2 import AdaptranV2
from ..config import (BufferConfig, OptimizerConfig, ReinforcedLearnerConfig,
                      SACAgentConfig)
from ..envs.wrappers.compatibility import (EnvProjectCompatibility,
                                           VecEnvProjectCompatibility)
from ..envs.wrappers.normalization import RMVNormalizeVecObservation
from ..utils import get_action_dim, get_observation_dim, set_torch_seed


def setup_env(
    env_cls_name,
    n_envs=1,
    seed=1,
    env_suite="carl",
    normalize_observation=False,
    # normalize_kwargs=None,
    render_mode=None,
):
    """Setup and return a vectorized environment."""

    def make_env():
        """Create and return an environment instance."""
        if env_suite == "carl":
            EnvCls: Type[
                Union[
                    CARLBraxHalfcheetah,
                    CARLDmcQuadrupedEnv,
                    CARLDmcWalkerEnv,
                    CARLMountainCarContinuous,
                    CARLPendulum,
                ]
            ] = eval(env_cls_name)

            print("Default context", EnvCls.get_default_context())
            # Just one context feature used
            obs_context_features = [list(EnvCls.get_context_features().keys())[0]]

            e: Union[
                CARLBraxHalfcheetah,
                CARLDmcQuadrupedEnv,
                CARLDmcWalkerEnv,
                CARLMountainCarContinuous,
                CARLPendulum,
            ] = EnvCls(
                obs_context_as_dict=False
            )  # obs_context_features=obs_context_features)

            # When CARL observation and action space inconsistent
            if isinstance(e.action_space, gym.Space) and isinstance(
                e.observation_space, gymz.Space
            ):
                e.action_space = _update_action_space(e)

            # * CARL SAC designed to handle Dict obs space
            # e = RecordEpisodeStatistics(
            #     FlattenObservation(FilterObservation(e, filter_keys=["obs"]))
            # )
        else:
            e = gymz.make(env_cls_name, render_mode=render_mode)
        
        if hasattr(e, "action_space"):
            e.action_space.seed(seed)
        return EnvProjectCompatibility(e)

    def _update_action_space(e):
        """Update action space of the environment to gymnasium spaces."""
        if isinstance(e, CARLDmcEnv):
            action_spec = e.env.env.action_spec()
            return gymz.spaces.Box(
                action_spec.minimum, action_spec.maximum, dtype=action_spec.dtype
            )

        return _convert_space(e.action_space)

    vec_env = gymz.vector.SyncVectorEnv([make_env for i in range(n_envs)])
    vec_env = VecEnvProjectCompatibility(vec_env)
    vec_env = RecordEpisodeStatistics(vec_env)
    vec_env = gymz.wrappers.StepAPICompatibility(vec_env, output_truncation_bool=False)

    # print("Seed: ", seed)
    # vec_env.reset(seed=seed)
    # vec_env.observation_space.seed(seed)
    # vec_env.action_space.seed(seed)
    return vec_env


if __name__ == "__main__":
    config = ReinforcedLearnerConfig(
        env_id="CARLMountainCarContinuous",
        batch_size=512,
        normalize_observation=False,
        agent=SACAgentConfig(
            name="exp-adaptranv2",
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

    agent = AdaptranV2(
        envs=envs,
        critic_learning_rate=config.agent.critic_optimizer.lr,
        actor_learning_rate=config.agent.actor_optimizer.lr,
        buffer_size=config.agent.buffer.buffer_size,
        device=config.device,
        writer=writer,
        log_freq=config.agent.log_freq,
        expanse_dim=config.agent.expanse_dim,
        seed=config.random_seed
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
    envs.close()
