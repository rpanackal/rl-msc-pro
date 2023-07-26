import collections
import numpy as np


def inspect_head_env(env, n_steps=1):
    # Verify the state, action, next state from environment
    obs = env.reset()
    print("0 Starting state: env.reset - ", obs)
    
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"{i} Action taken: env.action_space.sample() - ", action)
        print(f"{i+1} State: env.step - ", obs)


def inspect_head_dataset(dataset, n_steps=1):
    # Each task is associated with a dataset
    # dataset contains observations, actions, rewards, terminals, and infos

    for key, value in dataset.items():
        try:
            length = len(value)
        except Exception:
            length = None

        print(f"{key} type : {type(value)} length: {length}")
    
    print("0 Starting state: dataset['observations'][i] - ", dataset['observations'][0])
    for i in range(n_steps):
        print(f"{i} Action taken: dataset['actions'][i] - ", dataset['actions'][i])

        print(f"{i + 1} State: dataset['next_observations'][i]- ", dataset['next_observations'][i])
        print(f"{i + 1} State: dataset['observations'][i+1] - ", dataset['observations'][i+1])

def sequence_dataset(env, dataset=None, **kwargs):
    """
    Returns an iterator through trajectories.

    Args:
        env: An OfflineEnv object.
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

    N = dataset['rewards'].shape[0]
    
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for key, value in dataset.items():
            # We only retrive from values that span N items
            if type(value) is np.ndarray and len(value) == N:
                data_[key].append(value[i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1
