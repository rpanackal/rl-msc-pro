import warnings

warnings.simplefilter("ignore")

import json
from pprint import pprint
from typing import Dict

import gym
import gymnasium

from carl.envs import CARLBraxHalfcheetah, CARLCartPole
# from carl.envs.gymnasium import vec
from gym.envs import registry
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import half_cheetah

if __name__ == "__main__":
    # print("Gymnasium registry...")
    # gymnasium.pprint_registry()

    # print("Open AI Gym registry...")
    # pprint(gym.envs.registry.all())

    default_context = CARLCartPole.get_default_context()
    # new_context = default_context.copy()
    # new_context["length"] = new_context["length"] * 1.4

    env = CARLCartPole(obs_context_as_dict=False, obs_context_features=['length'])

    # print(env.spec)
    # print(f"Observation space ; {env.observation_space=}")
    # print(f"Action space ; {env.action_space=}")

    obs, _ = env.reset()
    # print(f"first obs {obs=}, length  {len(obs['obs'])}")
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample(), )

    print(f"{obs=}, type{type(obs)} length(obs)  {len(obs['obs'])} length(context) {len(obs['context'])}")
    print(f"{info=}")
    
    env = CARLCartPole(obs_context_as_dict=False)

    # print(env.spec)
    # print(f"Observation space ; {env.observation_space=}")
    # print(f"Action space ; {env.action_space=}")

    obs, _ = env.reset()
    # print(f"first obs {obs=}, length  {len(obs['obs'])}")
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample(), )

    print(f"{obs=}, type{type(obs)} length(obs)  {len(obs['obs'])} length(context) {len(obs['context'])}")
    print(f"{info=}")

    # From registry

    # env_2 = gymnasium.make("HalfCheetah-v2")
    # print(
    #     "Verify if Halfcheetah is registered in gymnasium: ",
    #     isinstance(env_2, CARLCartPole),
    # )

    # print(env_2.spec)
    # pprint(f"Observation space ; {env_2.observation_space=}")
    # pprint(f"Action space ; {env_2.action_space=}")

    # env_2.reset()
    # obs, reward, terminated, truncated, info = env_2.step(env_2.action_space.sample())

    # print(f"{obs=}")
    # print(f"{info=}")