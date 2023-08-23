from ..data.buffer import EpisodicBuffer

from datetime import datetime

import gym
import numpy as np
import torch

from ..config import ReinforcedLearnerConfig, SACAgentConfig
from ..envs.utils import make_env
from ..utils import set_torch_seed

def test_episodic_buffer(envs, config):
    rb = EpisodicBuffer(config.agent.buffer.buffer_size,
                        envs.single_observation_space,
                        envs.single_action_space,
                        config.device)
    
    obs = envs.reset()
    for _ in range(config.total_timesteps):
            # ALGO LOGIC: put action logic here
            # print("Training step: ", self.global_step)
            actions = np.array(
                [
                    envs.single_action_space.sample()
                    for _ in range(envs.num_envs)
                ]
            )

            # print(f"Actions in train, shape: {actions.shape}, type: {type(actions)}")
            next_obs, rewards, dones, infos = envs.step(actions)

            # print(f"Observations from env in train, shape: {next_obs.shape}, type: {type(next_obs)}")

            experience = (obs, next_obs, actions, rewards, dones, infos)
            rb.add(*experience)

            obs = next_obs

    samples = rb.sample(2, desired_length=100)
    print(f"Samples of batch 2 with desired length set: shape {samples.observations.shape} type {type(samples.observations)}")
    
    samples = rb.sample(2)
    print(f"Samples of batch 2: shape {len(samples.observations)} type {type(samples.observations)}")
    if isinstance(samples.observations, list):
        for sample in samples.observations:
            print(f"episode: shape {sample.shape} type {type(sample)}")


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

    test_episodic_buffer(envs, config)

    envs.close()
