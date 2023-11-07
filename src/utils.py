import collections
import numpy as np
import random
import torch
import carl
import typing
import gymnasium as gymz
import gym


def inspect_head_d4rl_env(env, n_steps=1):
    """Inspect head of D4RL registered environments

    Args:
        env (OfflineEnv): A registered offline environment
        n_steps (int, optional): Number of steps to print. Defaults to 1.
    """
    # Verify the state, action, next state from environment
    obs = env.reset()
    print("0 Starting state: env.reset - ", obs)

    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"{i} Action taken: env.action_space.sample() - ", action)
        print(f"{i+1} State: env.step - ", obs)


def inspect_head_d4rl_dataset(dataset, n_steps=1):
    """Inspect head of D4RL offline dataset

    Args:
        dataset (dict): A dictionary of keys like 'observations',
            'rewards', 'actions' etc..
        n_steps (int, optional): Number of steps to print. Defaults to 1.
    """
    # Each task is associated with a dataset
    # dataset contains observations, actions, rewards, terminals, and infos

    for key, value in dataset.items():
        try:
            length = len(value)
        except Exception:
            length = None

        print(f"{key} type : {type(value)} length: {length}")

    print("0 Starting state: dataset['observations'][i] - ", dataset["observations"][0])
    for i in range(n_steps):
        print(f"{i} Action taken: dataset['actions'][i] - ", dataset["actions"][i])

        print(
            f"{i + 1} State: dataset['next_observations'][i]- ",
            dataset["next_observations"][i],
        )
        print(
            f"{i + 1} State: dataset['observations'][i+1] - ",
            dataset["observations"][i + 1],
        )


def sequence_d4rl_dataset(env, dataset=None, **kwargs):
    """
    Returns an iterator through trajectories of D4RL
    directed dataset.

    Args:
        env (OfflineEnv): An gym registered offile environment.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset["rewards"].shape[0]

    data_ = collections.defaultdict(list)

    use_timeouts = "timeouts" in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset["terminals"][i])
        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1

        for key, value in dataset.items():
            # We only retrive from values that span N items
            if type(value) is np.ndarray and len(value) == N:
                data_[key].append(value[i])

        if done_bool or final_timestep:
            episode_step = 0
            yield {k: np.array(data_[k]) for k in data_}
            data_ = collections.defaultdict(list)

        episode_step += 1


def sequence_d4rl_dataset_numpy(self, env, split_length=None):
    """Collect all trajectories in a directed D4RL dataset into
    a numpy array.

    Args:
        env (OfflineEnv): An gym registered offile environment.
        split_length (_type_, optional): Split episodes into sequences of
            length split_length. Defaults to None.

    Raises:
        ValueError: Episode length is not divisible by split length

    Returns:
        np.ndaray: An array
            shape: (-1, split_length, feat_dim)
    """
    key_features = ("observations", "actions", "rewards")

    prev_episode_length = None
    dataset = []

    for episode in sequence_d4rl_dataset(env):
        episode_length = len(episode["rewards"])

        # When split_length is None, episode lengths are preserved
        split_length = split_length or episode_length

        assert episode_length % split_length == 0, ValueError(
            f"Episode length \
            ({episode_length}) is not divisible by split length ({split_length})"
        )

        n_parts = episode_length // split_length
        # Split observations, actions and rewards along time dimension
        # (episode_length, feat_dim) -> (n_parts, split_length, feat_dim)
        split_episodes = [
            episode[key].reshape(n_parts, split_length, -1) for key in key_features
        ]

        # Split observations, actions and rewards along feature dimension
        dataset.append(np.concatenate(split_episodes, axis=2))

        if prev_episode_length and prev_episode_length != episode_length:
            raise ValueError("Episodes are of different lengths.")
        else:
            prev_episode_length = episode_length

    return np.concatenate(dataset, axis=0)


def set_torch_seed(seed, torch_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def is_vector_env(env) -> bool:
    return (
        isinstance(env, (gymz.vector.VectorEnv, gym.vector.VectorEnv))
        or hasattr(env.unwrapped, "single_observation_space")
        or getattr(env.unwrapped, "is_vector_env", False)
        or getattr(env, "num_envs", 1) > 1
    )

def get_space_dimension(space, env) -> int:
    """
    Calculate and return the dimension of the space.

    Parameters:
    space (Space): The space whose dimension is to be calculated.
    env (Env): The environment instance to infer if it's a contextual environment.

    Returns:
    int: The dimension of the space.
    """
    # Check if the space is None and raise an error if it is
    if space is None:
        raise ValueError("Space not found in environment.")

    return np.prod(space.shape).astype(int).item()


def get_observation_dim(env: typing.Union[gymz.Env, gymz.vector.VectorEnv]) -> int:
    """
    Get the observation dimension of an environment.

    Parameters:
    env (Env): The environment whose observation dimension is to be calculated.

    Returns:
    int: The observation dimension of the environment.
    """
    # Determine the appropriate observation space
    space = (
        env.single_observation_space
        if hasattr(env, "single_observation_space")
        else env.observation_space
    )

    # Infer whether the environment is contextual
    is_contextual_env = getattr(env, "is_contextual_env", False)

    # Calculate the dimension based on whether the environment is contextual
    if is_contextual_env:
        space = space["obs"]
    
    # Get the dimension of the observation space
    return get_space_dimension(space, env)


def get_action_dim(env: typing.Union[gymz.Env, gymz.vector.VectorEnv]) -> int:
    """
    Get the action dimension of an environment.

    Parameters:
    env (Env): The environment whose action dimension is to be calculated.

    Returns:
    int: The action dimension of the environment.
    """
    # Determine the appropriate action space
    space = (
        env.single_action_space
        if hasattr(env, "single_action_space")
        else env.action_space
    )

    # Get the dimension of the action space
    return get_space_dimension(space, env)
