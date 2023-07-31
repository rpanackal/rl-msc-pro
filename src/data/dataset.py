# from torchvision import datasets
# from torchvision.transforms import ToTensor
import math

import gym
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils import sequence_dataset


class D4RLSequenceDataset(Dataset):
    key_features = ("observations", "actions", "rewards")

    def __init__(
        self,
        id: str,
        source_ratio: float = 0.7,
        transform=None,
        source_transform=None,
        target_transform=None,
        split_length=None,
    ):
        """_summary_

        Args:
            id (str): id string of d4rl dataset
            source_ratio (float, optional): The fraction of a episode to be kept
                be the source sequence. (1 - source_ratio) would be the fraction
                of target sequence. Defaults to 0.7.
            transform (_type_, optional): Transforms to be applied to whole sequence.
                Defaults to None.
            source_transform (_type_, optional): Transforms to be applied to source sequence.
                Defaults to None.
            target_transform (_type_, optional): Transforms to be applied to source sequence.
                Defaults to None.
            split_length (_type_, optional): If specified splits a sequence to create new 
                sequences of length split_length. Defaults to None.
        """
        assert 0 < source_ratio < 1, ValueError(
            "The source_ratio is not in range (0,1)"
        )

        self.env = gym.make(id)

        self.split_length = split_length
        self.source_ratio = source_ratio
        self.transform = transform
        self.source_transform = source_transform
        self.target_transform = target_transform

        self._dataset = self._get_sequence_dataset()

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        episode = self._dataset[idx]
        
        # self._dataset is list of ndarrays if 
        # self.is_variable_length is True, otherwise
        # self._dataset is a torch.Tensor
        if not isinstance(episode, torch.Tensor):
            episode = torch.Tensor(episode)

        if self.transform:
            episode = self.transform(episode)

        # Important to measure partition point after any transformation
        seq_length = episode.size(0)
        partition = math.floor(seq_length * self.source_ratio)

        history, target = (
            episode[:partition],
            episode[partition:],
        )

        if self.source_transform:
            history = self.source_transform(history)
        if self.target_transform:
            target = self.target_transform(target)
        return history, target

    def _get_sequence_dataset(self):
        dataset = []

        prev_episode_length = None
        is_variable_length = False

        for episode in sequence_dataset(self.env):
            episode_length = len(episode["rewards"])

            if prev_episode_length and prev_episode_length != episode_length:
                if self.split_length:
                    raise ValueError(
                        "Episodes of different lengths cannot be split. Set split_length to None"
                    )
            else:
                is_variable_length = True

            # When split_length is None, episode lengths are preserved
            split_length = self.split_length or episode_length

            # When episodes are equal length and split planned, run divisibility check
            assert episode_length % split_length == 0, ValueError(
                f"Episode length ({episode_length}) is not divisible by split length ({split_length})"
            )

            n_parts = episode_length // split_length
            # Split observations, actions and rewards along time dimension
            # (episode_length, feat_dim) -> (n_parts, split_length, feat_dim)
            split_episodes = [
                episode[key].reshape(n_parts, split_length, -1)
                for key in self.key_features
            ]

            # Concatenate observations, actions and rewards along feature dimension
            split_episodes = list(np.concatenate(split_episodes, axis=-1))
            dataset.extend(split_episodes)

            prev_episode_length = episode_length

        self.is_variable_length = is_variable_length
        return dataset if self.is_variable_length else torch.Tensor(dataset)

    def __getattr__(self, name):
        # Delegate attribute access to the true object
        return getattr(self._dataset, name)


if __name__ == "__main__":
    dataset = D4RLSequenceDataset("halfcheetah-medium-v2")
    source, target = dataset[0]
    print("Sequence in dataset index 0, shape: ", source.size(), target.size())
