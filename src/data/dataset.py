# from torchvision import datasets
# from torchvision.transforms import ToTensor
import math
import numpy as np
from d4rl.offline_env import OfflineEnv
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils import sequence_d4rl_dataset


class D4RLSequenceDataset(Dataset):
    feature_keys = ("observations", "actions", "rewards")

    def __init__(
        self,
        env: OfflineEnv,
        source_ratio: float = 0.7,
        transform=None,
        source_transform=None,
        target_transform=None,
        split_length=None,
        src_features_keys: list | None = None,
        tgt_features_keys: list | None = None
    ):
        """
        Initialize a dataset for D4RL environments with sequence data.

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
            src_features_keys (tuple | None, optional): Features of source sequence. Defaults to None.
            tgt_features_keys (tuple | None, optional): Features of target sequence. Defaults to None.
        """
        assert 0 < source_ratio < 1, ValueError(
            "The source_ratio is not in range (0,1)"
        )

        self.env = env
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)
        env.observation_space.shape[0]
        
        self.split_length = split_length
        self.source_ratio = source_ratio
        self.transform = transform
        self.source_transform = source_transform
        self.target_transform = target_transform

        self.src_features_keys = src_features_keys if src_features_keys else D4RLSequenceDataset.feature_keys
        for key in src_features_keys:
            if key not in D4RLSequenceDataset.feature_keys:
                raise ValueError("Unrecognized source feature keys given.")

        self.tgt_features_keys = tgt_features_keys if tgt_features_keys else D4RLSequenceDataset.feature_keys
        for key in tgt_features_keys:
            if key not in D4RLSequenceDataset.feature_keys:
                raise ValueError(f"Unrecognized target feature ({key}) keys given.")

        self._dataset = self._get_sequence_dataset()

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx: int):
        """_summary_

        Args:
            idx (int): Index of sequence in dataset.

        Returns:
            tuple[Tensor, Tensor, dict]: Returns source sequence, target sequence and extras.
                The extras is a contains features that were excluded from target sequence.
        """
        episode = self._dataset[idx]
        
        # self._dataset is list of ndarrays if self.is_variable_length is True, 
        # otherwise self._dataset is a torch.Tensor
        if not isinstance(episode, torch.Tensor):
            episode = torch.Tensor(episode)

        # Apply transform if any.
        if self.transform:
            episode = self.transform(episode)

        # Calculate partition index based on source_ratio after any transformation
        seq_length = episode.size(0)
        partition = math.floor(seq_length * self.source_ratio)

        source, target = episode[:partition], episode[partition:]
        source, _ = self.filter_features(source, self.src_features_keys)
        target, extras = self.filter_features(target, self.tgt_features_keys)

        # Apply source and target-specific transforms if any.
        if self.source_transform:
            source = self.source_transform(source)
        if self.target_transform:
            target = self.target_transform(target)
        return source, target, extras

    def _get_sequence_dataset(self):
        """Generate dataset of sequences.

        Raises:
            ValueError: When split_length given, but episodes in dataset are variable length.

        Returns:
            list | torch.Tensor: If sequences are same length then return a Tensor object, else a list.
                shape: (dataset_size, seq_len, feat_dim) 
        """
        dataset = []

        # Keep track of previous episode length and if sequences are of varying length
        prev_episode_length = None
        is_variable_length = False

        # Loop through each episode in d4rl dataset
        for episode in sequence_d4rl_dataset(self.env):

            # Calculate length of current episode
            episode_length = len(episode["rewards"])

            # If episode length are not consistent then set is_variable_length flag
            if prev_episode_length and prev_episode_length != episode_length:
                if self.split_length:
                    raise ValueError(
                        "Episodes of different lengths cannot be split. Set split_length to None"
                    )
                is_variable_length = True

            # When split_length is None, episode lengths are preserved
            split_length = self.split_length or episode_length

            # When episodes are equal length and split_length given, check if splitting possible
            assert episode_length % split_length == 0, ValueError(
                f"Episode length ({episode_length}) is not divisible by split length ({split_length})"
            )

            n_parts = episode_length // split_length
            # Split observations, actions and rewards along time dimension.
            # (episode_length, feat_dim) -> (n_parts, split_length, feat_dim)
            # If split_length is episdoe length, then no effective splitting.
            split_episodes = [
                episode[key].reshape(n_parts, split_length, -1)
                for key in self.feature_keys
            ]

            # Concatenate observations, actions and rewards along feature dimension
            split_episodes = list(np.concatenate(split_episodes, axis=-1))
            dataset.extend(split_episodes)

            # Update previous episode length
            prev_episode_length = episode_length

        self.is_variable_length = is_variable_length
        return dataset if self.is_variable_length else torch.Tensor(np.array(dataset))

    def filter_features(self, x: torch.FloatTensor, features: list[str]):
        """A function that preserve input features based on a passed tuple of features to keep.

        Args:
            x (torch.FloatTensor): Input tensor.
                shape: (seq_length, feat_dim)
            features (tuple[str]): Alowed feature values in D4RLSequenceDataset.feature_keys

        Returns:
            tuple[torch.FloatTensor, dict]: A tuple for filtered in input and a dictionary of
                discarded features.
        """
        extras = {}
        
        split = list(x.split([self.observation_dim, self.action_dim, 1], -1))

        for idx in reversed(range(len(split))):
            feature = D4RLSequenceDataset.feature_keys[idx]
            if  feature not in features:
                extras[feature] = split.pop(idx)

        x = torch.cat(split, dim=-1)
        return x, extras


    def __getattr__(self, name):
        # Delegate attribute access to the true object
        return getattr(self._dataset, name)

if __name__ == "__main__":
    dataset = D4RLSequenceDataset("halfcheetah-medium-v2")
    source, target = dataset[0]
    print("Sequence in dataset index 0, shape: ", source.size(), target.size())
