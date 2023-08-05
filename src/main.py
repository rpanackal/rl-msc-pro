# import gymnasium as gym
import d4rl  # Import required to register environments, you may need to also import the submodule
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from d4rl import gym_mujoco
from pydantic_settings import BaseSettings
from torch.utils.data import DataLoader, random_split

from .assets import Autoformer
from .core.config import AutoformerConfig, D4RLDatasetConfig, ExperimentConfig
from .data.dataset import D4RLSequenceDataset
from .learning.trainer import Trainer
from .transforms import RandomCropSequence


def d4rl_halfcheetah_pipeline():
    config = ExperimentConfig()
    config.model = AutoformerConfig()
    config.dataset = D4RLDatasetConfig(id="halfcheetah-medium-v2")

    # transform = transforms.Compose([RandomCropSequence(config.dataset.crop_length)])
    transform = None
    dataset = D4RLSequenceDataset(
        config.dataset.id,
        source_ratio=config.dataset.source_ratio,
        transform=transform,
        split_length=config.dataset.split_length,
    )

    train_dataset, valid_datset = random_split(
        dataset, [1 - config.dataset.validation_ratio, config.dataset.validation_ratio]
    )

    source, target = train_dataset[0]
    src_seq_length, feat_dim = source.size()
    tgt_seq_length = target.size(0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=config.dataloader.shuffle,
    )
    valid_loader = DataLoader(
        valid_datset,
        batch_size=config.dataloader.batch_size,
        shuffle=config.dataloader.shuffle,
    )

    model = Autoformer(
        feat_dim=feat_dim,
        embed_dim=config.model.embed_dim,
        expanse_dim=config.model.expanse_dim,
        kernel_size=config.model.kernel_size,
        corr_factor=config.model.corr_factor,
        n_enc_blocks=config.model.n_enc_blocks,
        n_dec_blocks=config.model.n_dec_blocks,
        n_heads=config.model.n_heads,
        src_seq_length=src_seq_length,
        tgt_seq_length=tgt_seq_length,
        cond_prefix_frac=config.model.cond_prefix_frac,
        dropout=config.model.dropout,
    ).to(config.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr)

    trainer = Trainer(
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
    )

    # Training
    for _ in range(config.n_epochs):
        result = trainer.step()
        if result.incumbent_found:
            trainer.save_checkpoint(result)


def main():
    d4rl_halfcheetah_pipeline()


if __name__ == "__main__":
    main()
