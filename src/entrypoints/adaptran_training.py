"""
Training logic for coretic V3 agent that used a pretrained frozen variational autoencoder
as represetnation model.
"""
from datetime import datetime
from typing import Union, Type

import gym
import gymnasium as gymz
import numpy as np
import torch
from carl.envs import (
    CARLBraxHalfcheetah,
    CARLDmcQuadrupedEnv,
    CARLDmcWalkerEnv,
    CARLMountainCarContinuous,
    CARLPendulum,
)
from carl.envs.dmc.carl_dmcontrol import CARLDmcEnv
from gymnasium.wrappers import (
    FilterObservation,
    FlattenObservation,
    RecordEpisodeStatistics,
)
from shimmy.openai_gym_compatibility import _convert_space
from torch.utils.tensorboard import SummaryWriter

from ..agents import Adaptran
from ..agents.adaptran import get_action_dim, get_observation_dim
from ..assets import Transformer, VariationalTransformer
from ..config import (
    BufferConfig,
    CoretranAgentConfig,
    OptimizerConfig,
    ReinforcedLearnerConfig,
    TransformerConfig,
    VariationalTransformerConfig,
)
from ..envs.core import make_env
from ..envs.wrappers.compatibility import (
    EnvProjectCompatibility,
    VecEnvProjectCompatibility,
)
from ..envs.wrappers.normalization import RMVNormalizeVecObservation
from ..utils import set_torch_seed


def load_pretrained_model(model: torch.nn.Module, path: str):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded pretrained model")
    return model


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

            # Just one context feature used
            obs_context_features = [list(EnvCls.get_context_features().keys())[0]]

            e: Union[
                CARLBraxHalfcheetah,
                CARLDmcQuadrupedEnv,
                CARLDmcWalkerEnv,
                CARLMountainCarContinuous,
                CARLPendulum,
            ] = EnvCls(
                obs_context_as_dict=False, obs_context_features=obs_context_features
            )

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
        
        # Seeding
        e.reset(seed=seed)
        if hasattr(e, "action_space"):
            e.action_space.seed(seed)
        if hasattr(e, "observation_space"):
            e.observation_space.seed(seed)
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
    vec_env = RMVNormalizeVecObservation(
        vec_env, is_observation_scaling=normalize_observation
    )
    vec_env = gymz.wrappers.StepAPICompatibility(vec_env, output_truncation_bool=False)
    return vec_env


def main():
    config = ReinforcedLearnerConfig(
        env_id="CARLMountainCarContinuous",
        batch_size=512,
        normalize_observation=True,
        agent=CoretranAgentConfig(
            name="adaptran-test",
            repr_model=TransformerConfig(
                embed_dim=8,
                n_enc_blocks=2,
                n_dec_blocks=1,
                src_seq_length=5,
                tgt_seq_length=5,
                cond_prefix_frac=0,
                load_from_path="",
                n_heads=2,
                # load_from_path="/home/rajanro/projects/rl-msc-pro/src/checkpoints/halfcheetah-expert-v2_transformer_2023-10-04_21-00-38/configuration.json",
            ),
            log_freq=100,
            repr_model_optimizer=OptimizerConfig(lr=1e-4),
            actor_optimizer=OptimizerConfig(lr=5e-4),
            critic_optimizer=OptimizerConfig(lr=5e-4),
            kappa=0.001,
            state_seq_length=2,
            target_network_frequency=8,
            policy_frequency=32,
            tau=0.01,
            autotune=False,
            alpha=0.1,
            gamma=0.9999,
            buffer=BufferConfig(buffer_size=5e4),
            expanse_dim=64,
        ),
        total_timesteps=1e5,
        device=torch.device("cuda:4" if torch.cuda.is_available() else "cpu"),
        n_envs=1,
        learning_starts=7e3,
        random_seed=60,
    )
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.name = f"{config.env_id}_{config.agent.name}_{current_datetime}"

    set_torch_seed(config.random_seed)

    envs: gymz.vector.SyncVectorEnv = setup_env(
        env_cls_name=config.env_id,
        n_envs=config.n_envs,
        seed=config.random_seed,
        # env_suite=args.package,
        normalize_observation=config.normalize_observation,
    )

    print("Device in use: ", config.device)
    print("Number of environments: ", envs.num_envs)
    print("Observation Space: ", envs.single_observation_space["obs"])
    print("Conetxt Space: ", envs.single_observation_space["context"])
    print("Action Space: ", envs.single_observation_space)

    obs_context_dim: dict = get_observation_dim(envs)
    observation_dim, context_dim = (
        obs_context_dim["obs"],
        obs_context_dim["context"],
    )
    action_dim = get_action_dim(envs)
    print("Source feature dimension", (observation_dim + action_dim))
    # config.agent.repr_model.head_dim = [context_dim]

    # Heuristic: Adapt embedding dimension to source dim

    # Embedding dim is ~75% of source_dim that is divisible by n_heads
    # quotient = (observation_dim + action_dim) * 0.75 // config.agent.repr_model.n_heads
    # config.agent.repr_model.embed_dim = int(
    #     config.agent.repr_model.n_heads * quotient
    # ) or (observation_dim + action_dim)
    # print("Embedding dimension", config.agent.repr_model.embed_dim)

    config.agent.repr_model.head_dims = [
        np.prod(envs.single_observation_space["context"].shape).item()
    ]

    # Define Representation Model
    # ! Important: Reconstruction of complete source
    model = Transformer(
        src_feat_dim=observation_dim + action_dim,
        tgt_feat_dim=observation_dim + action_dim,
        embed_dim=config.agent.repr_model.embed_dim,
        expanse_dim=config.agent.repr_model.expanse_dim,
        n_enc_blocks=config.agent.repr_model.n_enc_blocks,
        n_dec_blocks=config.agent.repr_model.n_dec_blocks,
        n_heads=config.agent.repr_model.n_heads,
        src_seq_length=config.agent.repr_model.src_seq_length,
        tgt_seq_length=config.agent.repr_model.tgt_seq_length,
        cond_prefix_frac=config.agent.repr_model.cond_prefix_frac,
        dropout=config.agent.repr_model.dropout,
        head_dims=config.agent.repr_model.head_dims,
    ).to(config.device)

    print(model)

    # Load pretrained model if present
    if config.agent.repr_model.load_from_path:
        model = load_pretrained_model(
            model,
            path=config.agent.repr_model.load_from_path,
        )

    # Setup trial logging
    log_dir = config.checkpoint_dir / config.name
    writer = SummaryWriter(log_dir=log_dir)

    config_path = log_dir / "configuration.json"
    with open(config_path, "w") as config_file:
        config_file.write(config.model_dump_json(exclude={"device"}))

    agent = Adaptran(
        envs=envs,
        repr_model=model,
        repr_model_learning_rate=config.agent.repr_model_optimizer.lr,
        critic_learning_rate=config.agent.critic_optimizer.lr,
        actor_learning_rate=config.agent.actor_optimizer.lr,
        buffer_size=config.agent.buffer.buffer_size,
        device=config.device,
        writer=writer,
        log_freq=config.agent.log_freq,
        expanse_dim=config.agent.expanse_dim,
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
        state_seq_length=config.agent.state_seq_length,
        kappa=config.agent.kappa,
        # kl_weight=config.agent.repr_model.kl_weight,
    )

    envs.close()


if __name__ == "__main__":
    main()
