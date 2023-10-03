import warnings

warnings.simplefilter("ignore")

import json
from pprint import pprint
from typing import Dict

import gym
import gymnasium

from carl.envs import CARLBraxHalfcheetah, CARLCartPole
from carl.envs.gymnasium import vec
from gym.envs import registry
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import half_cheetah

if __name__ == "__main__":
    # print("Gymnasium registry...")
    # gymnasium.pprint_registry()

    # print("Open AI Gym registry...")
    # pprint(gym.envs.registry.all())

    default_context = CARLCartPole.get_default_context()
    new_context = default_context.copy()
    new_context["length"] = new_context["length"] * 1.4

    env = CARLBraxHalfcheetah(obs_context_as_dict=True)

    # print(env.spec)
    # print(f"Observation space ; {env.observation_space=}")
    # print(f"Action space ; {env.action_space=}")

    obs, _ = env.reset()
    # print(f"first obs {obs=}, length  {len(obs['obs'])}")
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample(), )

    print(f"{obs=}, type{type(obs)} length(obs)  {len(obs['obs'])} length(context) {len(obs['context'])}")
    # print(f"{info=}")

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

    # pprint("CartPole-v0" in gym.envs.registry.env_specs)
    # print("#" * 10)
    # pprint(gymnasium.envs.registry)
    obs = {
        "obs": array(
            [-0.09044855, -0.02830288, -0.07037634, 0.00183157], dtype=float32
        ),
        "context": {
            "gravity": 9.8,
            "masscart": 1.0,
            "masspole": 0.1,
            "length": 0.5,
            "force_mag": 10.0,
            "tau": 0.02,
            "initial_state_lower": -0.1,
            "initial_state_upper": 0.1,
        },
    }
    info = {"context_id": 0}
