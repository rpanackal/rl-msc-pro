import math
from pathlib import PurePath
from typing import Union

import numpy as np
import torch
from gymnasium import Env
from torch.utils.data import Dataset

from ..utils import episodic_buffer_iterator


class EpisodicBufferDataset(Dataset):
    # TODO: Currently assumes context always exist as it is strictl present in feature_keys
    feature_keys = ("observations", "contexts", "actions", "rewards")

    def __init__(
        self,
        path: PurePath,
        source_ratio: float = 0.7,
        transform=None,
        source_transform=None,
        target_transform=None,
        split_length=None,
        src_features_keys: Union[list, None] = None,
        tgt_features_keys: Union[list, None] = None,
        do_normalize: bool = True,
        epsilon: float = 1e-8,
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
                sequences of length split_length. If the split_length is specified, it
                discards initial timesteps in each episode to ensure that the
                remaining length is divisible by split_length. Defaults to None.
            src_features_keys (Union[tuple, None], optional): Features of source sequence. Defaults to None.
            tgt_features_keys (Union[tuple, None], optional): Features of target sequence. Defaults to None.
            epsilon (float) : Used in the denominator of z-score normalization for stability.
        """
        assert 0 < source_ratio <= 1, ValueError(
            "The source_ratio is not in range (0,1)"
        )

        self.path = path if isinstance(path, PurePath) else PurePath(path)
        self.split_length = split_length
        self.source_ratio = source_ratio
        self.transform = transform
        self.source_transform = source_transform
        self.target_transform = target_transform

        self._feature_dims = None

        self.src_features_keys = (
            src_features_keys
            if src_features_keys
            else EpisodicBufferDataset.feature_keys
        )
        print("src_feat_keys", self.src_features_keys)
        for key in self.src_features_keys:
            if key not in EpisodicBufferDataset.feature_keys:
                raise ValueError(f"Unrecognized source feature keys given -{key}.")

        self.tgt_features_keys = (
            tgt_features_keys
            if tgt_features_keys
            else EpisodicBufferDataset.feature_keys
        )
        for key in self.tgt_features_keys:
            if key not in EpisodicBufferDataset.feature_keys:
                raise ValueError(f"Unrecognized target feature ({key}) keys given.")

        self._dataset = self._get_sequence_dataset()
        if do_normalize:
            self._dataset = self.normalize(self._dataset, epsilon)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx: int):
        """_summary_

        Args:
            idx (int): Index of sequence in dataset.

        Returns:
            tuple[Tensor, Tensor, dict]: Returns source sequence, target sequence and extras.
                The extras contains features that were excluded from source and target sequence.
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

        # TODO: Manage source and target if source ratio is 1
        source, target = episode[:partition], episode[partition:]

        # Apply source and target-specific transforms if any.
        extras = {}
        source, extras["source"] = self.filter_features(source, self.src_features_keys)
        if self.source_transform:
            source = self.source_transform(source)
    
        if target.numel() == 0:  # target maybe empty if source_ratio is 1
            target, extras["target"] = self.filter_features(target, self.tgt_features_keys)
            if self.target_transform:
                target = self.target_transform(target)
        return source, target, extras

    def _get_sequence_dataset(self):
        """Generate dataset of sequences. This function iterates over episodes stored in a buffer, 
        reshaping and combining them into a dataset suitable for training. If split_length
        is specified, it discards initial timesteps in each episode to ensure that the
        remaining length is divisible by split_length.

        Raises:
            ValueError: When split_length given, but episodes in dataset are variable length.

        Returns:
            Union[list, torch.Tensor]: If sequences are same length then return a Tensor object, else a list.
                shape: (dataset_size, seq_len, feat_dim)
        """
        dataset = []

        # Keep track of previous episode length and if sequences are of varying length
        prev_episode_length = None
        is_variable_length = False

        # Loop through each episode in d4rl dataset
        for idx, episode in enumerate(episodic_buffer_iterator(self.path)):
            # Get feature dims from first episode
            if idx == 0:
                self._feature_dims = [
                    episode[feature].shape[-1]
                    for feature in EpisodicBufferDataset.feature_keys
                ]
                print("Feature dims", self._feature_dims)

            # Calculate length of current episode
            episode_length = len(episode["rewards"])

            # If episode length are not consistent then set is_variable_length flag
            if prev_episode_length and prev_episode_length != episode_length:
                is_variable_length = True

            # When split_length is None, episode lengths are preserved
            final_length = self.split_length or episode_length
            n_parts = episode_length // final_length

            # Calculate the start index to slice the episode if episode_length not divisible by traj_length
            start_idx = (
                episode_length % self.split_length
                if self.split_length is not None
                else 0
            )

            # Split observations, actions and rewards along time dimension.
            # (episode_length, feat_dim) -> (n_parts, split_length, feat_dim)
            # If split_length is episdoe length, then no effective splitting.
            split_episodes = [
                episode[key][start_idx:].reshape(n_parts, final_length, -1)
                for key in EpisodicBufferDataset.feature_keys
            ]

            # Concatenate observations, actions and rewards along feature dimension
            split_episodes = list(np.concatenate(split_episodes, axis=-1))
            dataset.extend(split_episodes)

            # Update previous episode length
            prev_episode_length = episode_length

        self.is_variable_length = is_variable_length
        return (
            dataset
            if self.is_variable_length and self.split_length is None
            else torch.Tensor(np.array(dataset))
        )

    def filter_features(self, x: torch.FloatTensor, features: list[str]):
        """A function that preserve input features based on a passed tuple of features to keep.

        Args:
            x (torch.FloatTensor): Input tensor.
                shape: (seq_length, feat_dim)
            features (tuple[str]): Alowed feature values in EpisodicBufferDataset.feature_keys

        Returns:
            tuple[torch.FloatTensor, dict]: A tuple for filtered in input and a dictionary of
                discarded features.
        """
        extras = {}
        split = list(x.split(self._feature_dims, -1))

        for idx in reversed(range(len(split))):
            default_feature = EpisodicBufferDataset.feature_keys[idx]
            if default_feature not in features:
                extras[default_feature] = split.pop(idx)

        x = torch.cat(split, dim=-1)
        return x, extras

    def __getattr__(self, name):
        # Delegate attribute access to the true object
        return getattr(self._dataset, name)

    @staticmethod
    def normalize(dataset: torch.Tensor, epsilon=1e-8):
        if not isinstance(dataset, torch.Tensor):
            raise ValueError("Normalizaiton for non tensor dataset is not implemented!")

        mean = torch.mean(dataset, dim=(0, 1))
        std = torch.std(dataset, dim=(0, 1))

        return (dataset - mean) / (std + epsilon)
