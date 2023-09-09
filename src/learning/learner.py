import os
from datetime import datetime
import json
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from colorama import Fore, init
from ..core.datamodels import LearningEpochResult
from torch.utils.tensorboard import SummaryWriter
from pathlib import PurePath
from typing import Callable

init(autoreset=True)


class SupervisedLearner:
    """A class that handles traing and validation during learning a model."""

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        optimizer: optim.Optimizer,
        config,
        train_loader: DataLoader | None = None,
        valid_loader: DataLoader | None = None,
        lr_scheduler: optim.lr_scheduler.LRScheduler | None = None,
        custom_to_model: Callable | None = None,
        custom_to_criterion: Callable | None = None,
        writer: SummaryWriter | None = None,
        log_freq: int = 10,
        checkpoint_dir: str | None = None,
    ) -> None:
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = self.config.device

        self.custom_to_model = custom_to_model
        self.custom_to_criterion = custom_to_criterion

        self.lr_scheduler = lr_scheduler
        self.log_freq = log_freq
        self.checkpoint_dir = checkpoint_dir

        self.writer = writer

        # When trial_dir not set, tensorboard log_dir used if available
        if not self.checkpoint_dir and self.writer:
            self.checkpoint_dir = PurePath(writer.get_logdir())

        self.incumbent_loss = float("inf")
        self._epoch = -1
        self._train_iter = -1
        self._valid_iter = -1

    def train_batch(
        self,
        batch,
    ):
        """A training iteration step.

        Args:
            batch (tuple): A batch for input and target data.
        Returns:
            tuple[torch.Tensor, float]:
                0: predictions of the model for input X_batch
                1: loss computed for batch
        """
        self._train_iter += 1
        
        if not self.model.training:  # Checks if the model is in training mode
            self.model.train()
    
        # resets the gradients after every batch
        self.optimizer.zero_grad()

        args, kwargs = self.to_model(batch)
        output = self.model(*args, **kwargs)

        # compute loss and backpropage the gradients
        args, kwargs = self.to_criterion(batch, output)
        loss = self.criterion(*args, **kwargs)
        loss.backward()

        # update the weights
        self.optimizer.step()

        if self._train_iter % self.log_freq == 0 and self.writer:
            self.writer.add_scalar(
                "train/iteration/loss", loss.item(), self._train_iter
            )
            self.writer.add_scalar(
                "learning_rate", self.optimizer.param_groups[0]["lr"], self._train_iter
            )
            print(
                f"\nTraining - Loss at iteration {self._train_iter}: {loss.item()}",
                end=" ",
            )

        if self.lr_scheduler:
            self.lr_scheduler.step()

        return output, loss

    def train_epoch(self):
        """Train the model for an epoch. Model weights are being updated ,and,
        loss of training is collected.

        Returns:
            float: The training loss for one epoch
        """
        assert self.train_loader is not None, ValueError(
            "train_loader required to use epoch level training api of Learner \
                class."
        )

        # initialize every epoch
        epoch_loss = 0
        datset_size = len(self.train_loader.dataset)
        n_batches = len(self.train_loader)

        # set the model in training phase
        self.model.train()

        for batch_id, batch in enumerate(self.train_loader):
            batch_size = self.train_loader.batch_size

            _, loss = self.train_batch(batch)

            if self._train_iter % self.log_freq == 0:
                current = (batch_id + 1) * batch_size
                print(
                    f"| Epoch: {self._epoch} | "
                    + f"Progress: [{current}/{datset_size}]",
                    end=" ",
                )

            # loss
            epoch_loss += loss.item()

        print()
        return epoch_loss / n_batches

    def validate_batch(
        self,
        batch,
    ):
        """A validation iteration step.

        Note: It does not perform
        Args:
            X_batch (torch.Tensor): A batch for input data
            y_batch (torch.Tensor): A batch of groundtruth
        Returns:
            tuple[torch.Tensor, float]:
                0: predictions of the model for input X_batch
                1: loss computed for batch
        """
        self._valid_iter += 1

        if self.model.training:  # Checks if the model is in training mode
            self.model.eval()

        args, kwargs = self.to_model(batch)
        # deactivates autograd
        with torch.no_grad():
            output = self.model(*args, **kwargs)

            # Compute loss
            args, kwargs = self.to_criterion(batch, output)
            loss = self.criterion(*args, **kwargs)

        if self._valid_iter % self.log_freq == 0:
            if self.writer:
                self.writer.add_scalar(
                    "validation/iteration/loss", loss.item(), self._valid_iter
                )
            print(
                f"\nValidation - Loss at iteration {self._valid_iter}: {loss.item()}",
                end=" ",
            )

        return output, loss

    def validate_epoch(self):
        """Validate the model for an epoch on validation set. The validation
        loss is collected.

        Returns:
            tuple: A tuple of length 2 with validation loss and accuracy for one epoch
        """
        assert self.valid_loader is not None, ValueError(
            "valid_loader required to use epoch level valiation api of Learner \
                 class."
        )

        # initialize every epoch
        epoch_loss = 0
        n_batches = len(self.valid_loader)
        datset_size = len(self.valid_loader.dataset)

        # deactivating dropout layers
        self.model.eval()

        for batch_id, batch in enumerate(self.valid_loader):
            batch_size = self.valid_loader.batch_size

            _, loss = self.validate_batch(batch)

            # keep track of loss
            epoch_loss += loss.item()

            if self._valid_iter % self.log_freq == 0:
                current = (batch_id + 1) * batch_size
                print(
                    f"| Epoch: {self._epoch} | "
                    + f"Progress: [{current}/{datset_size}]",
                    end=" ",
                )
        print()
        return epoch_loss / n_batches

    def epoch(self):
        """
        One epoch of learning, which involves a training epoch and
        a validation epoch.

        Returns:
            LearningEpochResult: A pydantic model of epoch results
        """
        self._epoch += 1

        # train the model
        train_loss = self.train_epoch()

        # validate the model
        valid_loss = self.validate_epoch()

        incumbent_found = valid_loss < self.incumbent_loss
        if incumbent_found:
            self.incumbent_loss = valid_loss
        
        if self.writer:
            self.writer.add_scalar("train/epoch/loss", train_loss, self._epoch)
            self.writer.add_scalar("validation/epoch/loss", valid_loss, self._epoch)

        print(
            Fore.GREEN
            + f"\nEpoch: {self._epoch} complete | "
            + f"Training Loss: {train_loss} | "
            + f"Validation Loss: {valid_loss} | "
            + f"Incumbent Loss: {self.incumbent_loss}"
        )

        return LearningEpochResult(
            **{
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "epoch": self._epoch,
                "incumbent_loss": self.incumbent_loss,
                "incumbent_found": incumbent_found,
            }
        )

    def to_model(self, batch):
        """
        Prepare the input for the model's forward pass.
        
        Args:
            batch: tuple
                The batch of data to be processed. It is assumed to be a tuple (X_batch, y_batch).
        
        Returns:
            args: list
                Positional arguments to pass to the model.
            kwargs: dict
                Keyword arguments to pass to the model.
        """
        # Check if a custom function is provided, and if so, use it
        if self.custom_to_model:
            return self.custom_to_model(self, batch)
        
        # Default behavior: unpack the batch and move to the device
        X_batch, _ = batch
        
        args = [X_batch.to(self.device)]
        kwargs = {}
        return args, kwargs

    def to_criterion(self, batch, output):
        """
        Prepare the inputs for the loss calculation.
        
        Args:
            batch: tuple
                The batch of data. It is assumed to be a tuple (X_batch, y_batch).
            output: tensor
                The output from the model's forward pass.
        
        Returns:
            args: list
                Positional arguments to pass to the loss function.
            kwargs: dict
                Keyword arguments to pass to the loss function.
        """
        # Check if a custom function is provided, and if so, use it
        if self.custom_to_criterion:
            return self.custom_to_criterion(self, batch, output)
        
        # Default behavior: unpack the batch and move the label to the device
        _, y_batch = batch
        
        args = [output, y_batch.to(self.device)]
        kwargs = {}
        return args, kwargs

    def save_checkpoint(self, result: dict):
        """To checkpoint model at any epoch of a trial by saving
        model.state_dict() and optimizer.state_dict().

        Args:
            path: Path to trial directory
        """
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            checkpoint_path = self.checkpoint_dir / "checkpoint.pth"
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            torch.save(checkpoint, str(checkpoint_path))

            config_path = self.checkpoint_dir / "configuration.json"
            with open(config_path, "w") as config_file:
                config_file.write(self.config.model_dump_json())

            result_path = self.checkpoint_dir / "result.json"
            with open(result_path, "w") as result_file:
                result_file.write(result.model_dump_json())
        else:
            raise ValueError("Checkpoint directory not defined.")

    def load_checkpoint(self, path: str):
        """To load model checkpoint at any epoch of a trial by loading
        model.state_dict() and optimizer.state_dict().

        Args:
            path (str): A path for the trial checkpoint.
        """
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
