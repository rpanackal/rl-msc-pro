import gym
import gymnasium as gymz
from gymnasium.wrappers.compatibility import EnvCompatibility
from .wrappers.normalization import RMVNormalizeVecObservation
from carl.envs.brax.carl_brax_env import CARLBraxEnv
from shimmy.openai_gym_compatibility import _convert_space


def make_env(
    env: str | gymz.Env,
    seed: int,
    n_envs: int,
    capture_video: bool,
    run_name: str,
    normalize_observation: bool
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
                    raise ValueError(f"Environment {env} is not available in both gym and gymz.")
        else:
            if isinstance(env, gym.Env):
                e = gymz.make("GymV26Environment-v0", env=env)
            # elif isinstance(env, CARLBraxEnv):
            #     e = gymz.make("GymV26Environment-v0", env=env)
            elif isinstance(env, gymz.Env):
                e = env
            else:
                raise ValueError("The environment needs to be either a from Open AI Gym or Farma Foundation Gymnasium.")

        # Seeding
        e.reset(seed=seed + idx)
        if hasattr(e, "action_space"):
            e.action_space.seed(seed + idx)
        if hasattr(e, "observation_space"):
            e.observation_space.seed(seed + idx)
        print(e.observation_space)
        if hasattr(e,'action_space') and isinstance(e.action_space, gym.Space):
            e.action_space = _convert_space(e.action_space)

        # Video capture
        if capture_video and idx == 0:
            e = gymz.wrappers.RecordVideo(e, f"videos/{run_name}")
        return e
   
    envs = gymz.vector.SyncVectorEnv([lambda: single_env(i) for i in range(n_envs)])
    envs = gymz.wrappers.RecordEpisodeStatistics(envs)    
    envs = RMVNormalizeVecObservation(envs, is_observation_scaling=normalize_observation)
    envs = gymz.wrappers.StepAPICompatibility(envs, output_truncation_bool=False)
    
    return envs

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
        normalize_observation=normalize_observation
    )
