# import gymnasium as gym
import d4rl  # Import required to register environments, you may need to also import the submodule
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from d4rl import gym_mujoco
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import gym
import json

from ..assets import VariationalAutoformer
from ..config import (
    VariationalAutoformerConfig,
    D4RLDatasetConfig,
    SupervisedLearnerConfig,
    OptimizerConfig,
    CosineAnnealingLRConfig,
    DataLoaderConfig,
)
from ..data.dataset import D4RLSequenceDataset
from ..learning import SupervisedLearner, SupervisedEvaluator
from ..transforms import RandomCropSequence
from ..utils import flatten_dict
from torch.utils.tensorboard import SummaryWriter


def main():
    # Configure experiment
    config = SupervisedLearnerConfig(
        n_epochs=50,
        model=VariationalAutoformerConfig(
            embed_dim=16,
            expanse_dim=512,
            n_enc_blocks=2,
            n_dec_blocks=1,
            corr_factor=3,
            kl_weight=0.7,
            cond_prefix_frac=0
        ),
        dataset=D4RLDatasetConfig(
            env_id="halfcheetah-expert-v2", split_length=10, normalize_observation=False
        ),
        dataloader=DataLoaderConfig(batch_size=128),
        optimizer=OptimizerConfig(
            lr=1e-3, scheduler=CosineAnnealingLRConfig(min_lr=1e-5)
        ),
    )
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.name = f"{config.dataset.name}_{config.model.name}_{current_datetime}"

    # Create Gym environment
    env = gym.make(config.dataset.name)

    # Handle dataset and dataloaders
    # transform = transforms.Compose([RandomCropSequence(config.dataset.crop_length)])
    transform = None
    dataset = D4RLSequenceDataset(
        env=env,
        source_ratio=config.dataset.source_ratio,
        transform=transform,
        split_length=config.dataset.split_length,
        src_features_keys=["observations", "actions"],
        tgt_features_keys=["observations"],
        do_normalize=config.dataset.normalize_observation,
    )

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset,
        [
            1 - config.dataset.validation_ratio - config.dataset.test_ratio,
            config.dataset.validation_ratio,
            config.dataset.test_ratio,
        ],
    )

    source, target, _ = train_dataset[0]
    config.model.src_seq_length, src_feat_dim = source.size()
    config.model.tgt_seq_length, tgt_feat_dim = target.size()

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
    # ! Source reconstruction, tgt_feat_dim replaced with src_feat_dim
    model = VariationalAutoformer(
        src_feat_dim=src_feat_dim,
        tgt_feat_dim=src_feat_dim,
        embed_dim=config.model.embed_dim,
        expanse_dim=config.model.expanse_dim,
        kernel_size=config.model.kernel_size,
        corr_factor=config.model.corr_factor,
        n_enc_blocks=config.model.n_enc_blocks,
        n_dec_blocks=config.model.n_dec_blocks,
        n_heads=config.model.n_heads,
        src_seq_length=config.model.src_seq_length,
        tgt_seq_length=config.model.tgt_seq_length,
        cond_prefix_frac=config.model.cond_prefix_frac,
        dropout=config.model.dropout,
        full_output=True,
    ).to(config.device)

    # Define optimizer and criteria
    def criterion(output, target, kl_weight=config.model.kl_weight):
        dec_output, _, mean, logvar, _ = output
        recon_loss = F.mse_loss(dec_output, target)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl_loss  # Now actually using kl_weight

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr)

    max_iter = config.n_epochs * (len(train_dataset) / config.dataloader.batch_size)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=config.optimizer.scheduler.min_lr, T_max=max_iter
    )

    # Setup trial logging
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
    result = evaluator.test()

    # writer.add_hparams(
    #     hparam_dict=flatten_dict(json.loads(config.model_dump_json()), sep="/"),
    #     metric_dict={"test/avg_loss": result.loss}, run_name="hyperparameters"
    # )
    env.close()
    writer.close()


# Soure reconstruction

def custom_to_model(learner, batch):
    source, _, extras = batch

    args = []
    kwargs = {
        "source": source.to(learner.device),
        "dec_init": None,
    }

    return args, kwargs


def custom_to_criterion(learner, batch, output):
    source, _, _ = batch
    
    target = source
    args = [output, target.to(learner.device)]
    kwargs = {}

    return args, kwargs

# Time series forecasting

# def custom_to_model(learner, batch):
#     source, _, extras = batch

#     args = []
#     kwargs = {
#         "source": source.to(learner.device),
#         "dec_init": extras["actions"].to(learner.device),
#     }

#     return args, kwargs


# def custom_to_criterion(learner, batch, output):
#     _, target, _ = batch

#     args = [output, target.to(learner.device)]
#     kwargs = {}

#     return args, kwargs


if __name__ == "__main__":
    main()
