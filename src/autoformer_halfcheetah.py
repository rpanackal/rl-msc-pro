# import gymnasium as gym
import d4rl  # Import required to register environments, you may need to also import the submodule
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from d4rl import gym_mujoco
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import gym

from .assets import Autoformer
from .core.config import AutoformerConfig, D4RLDatasetConfig, ExperimentConfig
from .data.dataset import D4RLSequenceDataset
from .learning import Learner, Evaluator
from .transforms import RandomCropSequence
from torch.utils.tensorboard import SummaryWriter


def main():

    # Configure experiment
    config = ExperimentConfig()
    config.model = AutoformerConfig()
    config.dataset = D4RLDatasetConfig(id="halfcheetah-medium-v2")

    # Create Gym environment
    env = gym.make(config.dataset.id)

    # Handle dataset and dataloaders
    # transform = transforms.Compose([RandomCropSequence(config.dataset.crop_length)])
    transform = None
    dataset = D4RLSequenceDataset(
        env=env,
        source_ratio=config.dataset.source_ratio,
        transform=transform,
        split_length=config.dataset.split_length,
    )

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset,
        [
            1 - config.dataset.validation_ratio - config.dataset.test_ratio,
            config.dataset.validation_ratio,
            config.dataset.test_ratio,
        ],
    )

    source, target = train_dataset[0]
    src_seq_length, src_feat_dim = source.size()
    tgt_seq_length, tgt_feat_dim = target.size()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=config.dataloader.shuffle,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=config.dataloader.shuffle,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=config.dataloader.shuffle,
    )

    # Define Model
    model = Autoformer(
        src_feat_dim=src_feat_dim,
        tgt_feat_dim=tgt_feat_dim,
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

    # Define optimizer and criteria
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr)

    max_iter = config.n_epochs * (len(train_dataset) / config.dataloader.batch_size)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=config.optimizer.min_lr, T_max=max_iter
    )

    # Setup trial logging
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    trial_name = f"{config.dataset.id}_{config.model.name}_{current_datetime}"
    log_dir = config.checkpoint_dir / trial_name
    writer = SummaryWriter(log_dir=log_dir)

    # Define learner
    learner = Learner(
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        writer=writer,
    )

    # Training
    for _ in range(config.n_epochs):
        result = learner.epoch()
        if result.incumbent_found:
            learner.save_checkpoint(result)

    evaluator = Evaluator(
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        config=config,
        writer=writer,
    )
    evaluator.test()

    env.close()
    writer.close()

if __name__ == "__main__":
    main()
