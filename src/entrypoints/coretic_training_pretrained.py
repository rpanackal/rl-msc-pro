"""
Training logic for coretic V3 agent that used a pretrained frozen variational autoencoder
as represetnation model.
"""
import time
from datetime import datetime

import gym
import numpy as np
import torch

from ..agents import CoreticAgent
from ..config import (
    ReinforcedLearnerConfig,
    CoreticAgentConfig,
    VariationalAutoformerConfig,
    AutoformerConfig,
    OptimizerConfig,
)
from ..envs.core import make_env
from ..utils import set_torch_seed
from ..assets import VariationalAutoformer, Autoformer
from ..envs.wrappers.normalization import RMVNormalizeVecObservation
from torch.utils.tensorboard import SummaryWriter


def load_pretrained_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded pretrained model")
    return model


if __name__ == "__main__":
    config = ReinforcedLearnerConfig(
        agent=CoreticAgentConfig(
            name="pretrained-coretic",
            repr_model=AutoformerConfig(
                embed_dim=16,
                n_enc_blocks=2,
                n_dec_blocks=1,
                src_seq_length=50,
                tgt_seq_length=50,
                corr_factor=3,
                # kl_weight=0.7,
                expanse_dim=512,
                cond_prefix_frac=0
            ),
            log_freq=100,
            repr_model_optimizer=OptimizerConfig(lr=1e-4),
            actor_optimizer=OptimizerConfig(lr=1e-4),
            critic_optimizer=OptimizerConfig(lr=1e-3),
            kappa=0.001,
            state_seq_length=2,
            target_network_frequency=2,
            policy_frequency=4,
            autotune=True,
            alpha=0.2
        ),
        total_timesteps=1e6,
        learning_starts=10000,
        batch_size=64,
        normalize_observation=False,
        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu"),
        n_envs=4
    )
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.name = f"{config.env_id}_{config.agent.name}_{current_datetime}"

    set_torch_seed(config.random_seed)

    # Here only 1 environment
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env=config.env_id,
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

    print("Device in use: ", config.device)
    print("Number of environments: ", envs.num_envs)
    print("Observation Space: ", envs.single_observation_space)
    print("Action Space: ", envs.single_action_space)

    observation_dim = np.prod(envs.single_observation_space.shape)
    action_dim = np.prod(envs.single_action_space.shape)

    # Define Representation Model
    # * Important: Reconstruction of complete source
    model = Autoformer(
        src_feat_dim=observation_dim + action_dim,
        tgt_feat_dim=observation_dim + action_dim,
        embed_dim=config.agent.repr_model.embed_dim,
        expanse_dim=config.agent.repr_model.expanse_dim,
        kernel_size=config.agent.repr_model.kernel_size,
        corr_factor=config.agent.repr_model.corr_factor,
        n_enc_blocks=config.agent.repr_model.n_enc_blocks,
        n_dec_blocks=config.agent.repr_model.n_dec_blocks,
        n_heads=config.agent.repr_model.n_heads,
        src_seq_length=config.agent.repr_model.src_seq_length,
        tgt_seq_length=config.agent.repr_model.tgt_seq_length,
        cond_prefix_frac=config.agent.repr_model.cond_prefix_frac,
        dropout=config.agent.repr_model.dropout,
    ).to(config.device)

    print(model)

    model = load_pretrained_model(
        model,
        path="/home/rajanro/projects/rl-msc-pro/src/checkpoints/halfcheetah-expert-v2_autoformer_2023-09-15_21-51-19/checkpoint.pth",
    )

    # Setup trial logging
    log_dir = config.checkpoint_dir / config.name
    writer = SummaryWriter(log_dir=log_dir)

    config_path = log_dir / "configuration.json"
    with open(config_path, "w") as config_file:
        config_file.write(config.model_dump_json(exclude={"device"}))

    agent = CoreticAgent(
        envs=envs,
        repr_model=model,
        repr_model_learning_rate=config.agent.repr_model_optimizer.lr,
        critic_learning_rate=config.agent.critic_optimizer.lr,
        actor_learning_rate=config.agent.actor_optimizer.lr,
        buffer_size=config.agent.buffer.buffer_size,
        device=config.device,
        writer=writer,
        log_freq=config.agent.log_freq,
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

    # agent.test(n_episodes=10)

    envs.close()
