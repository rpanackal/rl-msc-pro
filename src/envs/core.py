from typing import Type, Union

import gym
import gymnasium as gymz
import numpy as np
from carl.context.context_space import (
    NormalFloatContextFeature,
    UniformFloatContextFeature,
)
from carl.context.sampler import ContextSampler
from carl.envs import (
    CARLBraxHalfcheetah,
    CARLDmcQuadrupedEnv,
    CARLDmcWalkerEnv,
    CARLMountainCarContinuous,
    CARLPendulum,
    CARLVehicleRacing,
    CARLDmcFishEnv
)
from carl.envs.carl_env import CARLEnv
from carl.envs.dmc.carl_dmcontrol import CARLDmcEnv
from gymnasium.experimental.vector.utils import batch_space
from gymnasium.wrappers import (
    FilterObservation,
    FlattenObservation,
    RecordEpisodeStatistics,
)
from gymnasium.wrappers.compatibility import EnvCompatibility
from shimmy.openai_gym_compatibility import _convert_space

from .wrappers.batch_step import EnvVectorResponse
from .wrappers.compatibility import EnvProjectCompatibility, VecEnvProjectCompatibility
from .wrappers.normalization import RMVNormalizeVecObservation

# TODO: Merge make_env and setup_env to one


def make_env(
    env: Union[str, gymz.Env],
    seed: int,
    n_envs: int,
    capture_video: bool,
    run_name: str,
    normalize_observation: bool,
) -> gymz.vector.SyncVectorEnv:
    """
    Create a vectorized gymz environment wrapped with specified settings.

    Args:
        env (str): The ID of the environment to create.
        seed (int): The random seed to set for the environment.
        n_envs (int): Number of environments to create.
        capture_video (bool): Whether to capture videos of the environment.
        run_name (str): The name of the run, used for saving videos.
        normalize_observation (bool): Whether to normalize the observations.

    Returns:
        gymz.vector.SyncVectorEnv: The wrapped vectorized environment.
    """

    def single_env(idx: int) -> gymz.Env:
        if isinstance(env, str):
            # Attempt to create env from gymz
            try:
                e = gymz.make(env)
            except:
                if env in gym.envs.registry.env_specs:
                    e = gymz.make("GymV26Environment-v0", env_id=env)
                else:
                    raise ValueError(
                        f"Environment {env} is not available in both gym and gymz."
                    )
        else:
            if isinstance(env, gym.Env):
                e = gymz.make("GymV26Environment-v0", env=env)
            elif isinstance(env, gymz.Env):
                e = env
            else:
                raise ValueError(
                    "The environment needs to be either a from Open AI Gym or Farma Foundation Gymnasium."
                )

        e = EnvProjectCompatibility(e)
        # Seeding
        e.reset(seed=seed + idx)
        if hasattr(e, "action_space"):
            e.action_space.seed(seed + idx)
        if hasattr(e, "observation_space"):
            e.observation_space.seed(seed + idx)

        # Video capture
        if capture_video and idx == 0:
            e = gymz.wrappers.RecordVideo(e, f"videos/{run_name}")
        return e

    if n_envs > 1:
        envs = gymz.vector.SyncVectorEnv([lambda: single_env(i) for i in range(n_envs)])
        envs = VecEnvProjectCompatibility(envs)
        envs = gymz.wrappers.RecordEpisodeStatistics(envs)
        envs = RMVNormalizeVecObservation(
            envs, is_observation_scaling=normalize_observation
        )
        envs = gymz.wrappers.StepAPICompatibility(envs, output_truncation_bool=False)
        envs = gymz.wrappers.PassiveEnvChecker(envs)
    else:
        envs = single_env(0)
        envs = EnvVectorResponse(envs)
        envs = gymz.wrappers.AutoResetWrapper(envs)
        envs = gymz.wrappers.RecordEpisodeStatistics(envs)
        # envs = RMVNormalizeVecObservation(envs, is_observation_scaling=normalize_observation)
        envs = gymz.wrappers.StepAPICompatibility(envs, output_truncation_bool=False)
        envs = gymz.wrappers.PassiveEnvChecker(envs)

    return envs


context_distributions = {
    "CARLMountainCarContinuous": [
        # Small changes to the track's lower bound (min_position) can simulate different
        # starting slopes.
        NormalFloatContextFeature(
            "min_position", mu=-1.2, sigma=0.2, default_value=-1.2
        ),
        # Similar to min_position, this changes the track's upper bound and simulates different ending
        # slopes.
        NormalFloatContextFeature("max_position", mu=0.6, sigma=0.2, default_value=0.6),
        # Varying the maximum speed can simulate different friction or power conditions.
        NormalFloatContextFeature("max_speed", mu=0.07, sigma=0.02, default_value=0.07),
        # The goal position should be within a small range to ensure the goal is always reachable
        # but its exact position can vary slightly to change the challenge.
        UniformFloatContextFeature(
            "goal_position", lower=0.4, upper=0.6, default_value=0.5
        ),
        # Power affects how quickly the car can accelerate, simulating various engine strengths.
        NormalFloatContextFeature(
            "power", mu=0.0015, sigma=0.0002, default_value=0.0015
        ),
    ]
}


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
                    CARLVehicleRacing
                ]
            ] = eval(env_cls_name)

            print("Default context", EnvCls.get_default_context())
            # Just one context feature used
            obs_context_features = [list(EnvCls.get_context_features().keys())[0]]
            context_sampler = ContextSampler(
                context_distributions=context_distributions[env_cls_name],
                context_space=EnvCls.get_context_space(),
                seed=seed,
            )
            contexts = context_sampler.sample_contexts(n_contexts=100)
            print(contexts)

            e: Union[
                CARLBraxHalfcheetah,
                CARLDmcQuadrupedEnv,
                CARLDmcWalkerEnv,
                CARLMountainCarContinuous,
                CARLPendulum,
                CARLVehicleRacing
            ] = EnvCls(
                obs_context_as_dict=False,
                contexts=contexts
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
    vec_env = gymz.wrappers.RecordEpisodeStatistics(vec_env)
    vec_env = gymz.wrappers.StepAPICompatibility(vec_env, output_truncation_bool=False)

    return vec_env


if __name__ == "__main__":
    env_id = "CartPole-v1"
    random_seed = 10
    n_envs = 4
    run_name = "unit_test_make_env"
    capture_video = False
    normalize_observation = False

    envs = make_env(
        env=env_id,
        seed=random_seed,
        n_envs=n_envs,
        capture_video=capture_video,
        run_name=run_name,
        normalize_observation=normalize_observation,
    )
