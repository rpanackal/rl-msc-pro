# import gymnasium as gym
import d4rl  # Import required to register environments, you may need to also import the submodule
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from d4rl import gym_mujoco
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import gym

from ..config import (
    OrigAutoformerConfig,
    AutoformerConfig,
    D4RLDatasetConfig,
    SupervisedLearnerConfig,
    OptimizerConfig,
    CosineAnnealingLRConfig,
    DataLoaderConfig,
)
from ..data.dataset import D4RLSequenceDataset
from ..learning import SupervisedLearner, SupervisedEvaluator
from ..transforms import RandomCropSequence
from torch.utils.tensorboard import SummaryWriter
from ..external.Autoformer.models.Autoformer import Model as Autoformer


def main():
    # Configure experiment
    dataset_config = D4RLDatasetConfig(
        env_id="halfcheetah-medium-v2",
        split_length=10,
    )
    dataloader_config = DataLoaderConfig(batch_size=128)

    # Create Gym environment
    env = gym.make(dataset_config.name)

    # Handle dataset and dataloaders
    # transform = transforms.Compose([RandomCropSequence(config.dataset.crop_length)])
    transform = None
    dataset = D4RLSequenceDataset(
        env=env,
        source_ratio=dataset_config.source_ratio,
        transform=transform,
        split_length=dataset_config.split_length,
        src_features_keys=["observations", "actions"],
        tgt_features_keys=["observations"],
        do_normalize=True,
    )

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset,
        [
            1 - dataset_config.validation_ratio - dataset_config.test_ratio,
            dataset_config.validation_ratio,
            dataset_config.test_ratio,
        ],
    )

    source, target, extras = train_dataset[0]
    src_seq_length, src_feat_dim = source.size()
    tgt_seq_length, tgt_feat_dim = target.size()

    config = SupervisedLearnerConfig(
        n_epochs=15,
        model=OrigAutoformerConfig(
            factor=3,
            d_model=32,
            enc_in=src_feat_dim,
            dec_in=src_feat_dim,
            c_out=tgt_feat_dim,
            seq_len=src_seq_length,
            label_len=int(src_seq_length * 0.3),
            pred_len=tgt_seq_length,
            d_ff=512
        ),
        dataset=dataset_config,
        dataloader=dataloader_config,
        optimizer=OptimizerConfig(
            lr=0.0001, scheduler=CosineAnnealingLRConfig(min_lr=0.00001)
        ),
    )

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.name = f"{config.dataset.name}_{config.model.name}_{current_datetime}"

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
    model = Autoformer(configs=config.model).to(config.device)

    # Define optimizer and criteria
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr)

    max_iter = config.n_epochs * (len(train_dataset) / config.dataloader.batch_size)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=config.optimizer.scheduler.min_lr, T_max=max_iter
    )

    log_dir = config.checkpoint_dir / config.name
    writer = SummaryWriter(log_dir=log_dir)

    # Define learner
    learner = SupervisedLearner(
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        writer=writer,
        custom_to_model=custom_to_model,
        custom_to_criterion=custom_to_criterion,
    )

    # Training
    for _ in range(config.n_epochs):
        result = learner.epoch()
        if result.incumbent_found:
            learner.save_checkpoint(result)

    evaluator = SupervisedEvaluator(
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        config=config,
        writer=writer,
        custom_to_model=custom_to_model,
        custom_to_criterion=custom_to_criterion,
    )
    evaluator.test()

    env.close()
    writer.close()


def custom_to_model(learner, batch):
    source, target, extras = batch

    args = []
    kwargs = {
        "x_enc": source.to(learner.device),
        "x_dec": extras["actions"].to(learner.device)
        # "x_dec": target.to(learner.device),
    }

    return args, kwargs


def custom_to_criterion(learner, batch, output):
    _, target, _ = batch

    args = [output, target.to(learner.device)]
    kwargs = {}

    return args, kwargs


if __name__ == "__main__":
    main()
