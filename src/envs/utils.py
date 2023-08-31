from typing import Callable
import gym
from gym.wrappers import RecordEpisodeStatistics, RecordVideo

def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str) -> Callable:
    """
    Create a gym environment with specified wrappers and settings.

    Args:
        env_id (str): The ID of the gym environment to create.
        seed (int): The random seed to set for the environment.
        idx (int): The index of the environment, used to determine if videos should be captured.
        capture_video (bool): Whether to capture videos of the environment.
        run_name (str): The name of the run, used for saving videos.

    Returns:
    - Callable: A thunk that creates and returns the configured environment.
    """

    def thunk() -> gym.Env:
        env = gym.make(env_id)
        env = RecordEpisodeStatistics(env)

        if capture_video and idx == 0:
            env = RecordVideo(env, f"videos/{run_name}")

        if hasattr(env, "seed"):
            env.seed(seed)
        if hasattr(env, "action_space"):
            env.action_space.seed(seed)
        if hasattr(env, "observation_space"):
            env.observation_space.seed(seed)

        return env

    return thunk
