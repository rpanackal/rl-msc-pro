from datetime import datetime
import numpy as np
import torch
from carl.envs.dmc.carl_dmcontrol import CARLDmcEnv
from shimmy.openai_gym_compatibility import _convert_space
from torch.utils.tensorboard import SummaryWriter

from ..envs.core import setup_env

from ..agents.adaptran_v2 import AdaptranV2
from ..agents.adaptran_v3 import AdaptranV3
from ..agents.carl_sac import CARLSACAgent
from ..assets import Transformer
from ..config import (
    AdaptranAgentConfig,
    BufferConfig,
    OptimizerConfig,
    ReinforcedLearnerConfig,
    SACAgentConfig,
    TransformerConfig,
)
from ..utils import (
    compare_random_states,
    get_action_dim,
    get_observation_dim,
    get_random_states,
    set_torch_seed,
)


def load_pretrained_model(model: torch.nn.Module, path: str):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded pretrained model")
    return model


if __name__ == "__main__":
    config = ReinforcedLearnerConfig(
        env_id="CARLMountainCarContinuous",
        batch_size=32,
        normalize_observation=False,
        agent=AdaptranAgentConfig(
            name="spiked-adaptranv3",
            repr_model=TransformerConfig(
                embed_dim=4,
                n_enc_blocks=2,
                n_dec_blocks=2,
                src_seq_length=5,
                tgt_seq_length=5,
                cond_prefix_frac=0,
                # load_from_path="",
                n_heads=2,
                expanse_dim=32,
                load_from_path="/home/rajanro/projects/rl-msc-pro/src/checkpoints/CARLMountainCarContinuous-check_transformer_2023-11-30_01-12-53/checkpoint.pth",
            ),
            repr_model_optimizer=OptimizerConfig(lr=3e-4),
            actor_optimizer=OptimizerConfig(lr=3e-4),
            critic_optimizer=OptimizerConfig(lr=3e-4),
            target_network_frequency=1,
            policy_frequency=1,
            kappa=0.01,
            tau=0.01,
            autotune=False,
            alpha=0.2,
            gamma=0.9999,
            buffer=BufferConfig(buffer_size=1e5),
            expanse_dim=128,
            critic_gradient_prop=True,
        ),
        total_timesteps=1e5,
        device=torch.device("cuda:5" if torch.cuda.is_available() else "cpu"),
        n_envs=1,
        learning_starts=512 * (5 + 1),
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

    obs_context_dim: dict = get_observation_dim(envs)
    observation_dim, context_dim = (
        obs_context_dim["obs"],
        obs_context_dim["context"],
    )
    action_dim = get_action_dim(envs)
    print("Context space dimensionality: ", obs_context_dim)

    # Define Representation Model
    config.agent.repr_model.head_dims = [context_dim]

    initial_states = get_random_states()
    # ! Pure existence means chaos
    repr_model = Transformer(
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

    if config.agent.repr_model.load_from_path:
        model = load_pretrained_model(
            repr_model,
            path=config.agent.repr_model.load_from_path,
        )

    #! Resetting seed again is super necessary and can change the outcome of training
    set_torch_seed(config.random_seed)

    post_states = get_random_states()
    changes = compare_random_states(initial_states, post_states)
    for source, is_changed in changes.items():
        if is_changed:
            raise RuntimeError(
                f"Unintended random state change encounered in {source}!"
            )

    preloaded_replay_buffer_path = "/home/rajanro/projects/rl-msc-pro/src/checkpoints/CARLMountainCarContinuous_Adaptran-mod-sac_2023-11-22_17-01-20/buffer.npz"

    agent = AdaptranV3(
        envs=envs,
        repr_model=repr_model,
        repr_model_learning_rate=config.agent.repr_model_optimizer.lr,
        critic_learning_rate=config.agent.critic_optimizer.lr,
        actor_learning_rate=config.agent.actor_optimizer.lr,
        buffer_size=config.agent.buffer.buffer_size,
        device=config.device,
        writer=writer,
        log_freq=config.agent.log_freq,
        expanse_dim=config.agent.expanse_dim,
        seed=config.random_seed,
        critic_gradient_prop=config.agent.critic_gradient_prop,
        preloaded_replay_buffer_path=preloaded_replay_buffer_path,
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
