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


init(autoreset=True)


class SupervisedLearner:
    """A class that handles traing and validation during learning a model."""

    def __init__(
        self,
        model: nn.Module,
        criterion,
        optimizer: optim.Optimizer,
        config,
        train_loader: DataLoader | None = None,
        valid_loader: DataLoader | None = None,
        lr_scheduler: optim.lr_scheduler.LRScheduler | None = None,
        writer: SummaryWriter | None = None,
        log_freq: int = 10,
        trial_dir: str | None = None,
    ) -> None:
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config

        self.lr_scheduler = lr_scheduler
        self.log_freq = log_freq
        self.trial_dir = trial_dir

        # When trial_dir not set
        if not self.trial_dir:
            # Trial directory same as tensorboard log_dir if available
            if writer:
                self.trial_dir = PurePath(writer.get_logdir())

            # Create a trial dir from config information and timestamp
            else:
                current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                trial_name = f"{self.config.dataset.name}_{self.config.model.name}_{current_datetime}"
                self.trial_dir = self.config.checkpoint_dir / trial_name

        self.writer = writer if writer else SummaryWriter(log_dir=str(self.trial_dir))

        self.incumbent_loss = float("inf")
        self._epoch = -1
        self._train_iter = -1
        self._valid_iter = -1

    def train_batch(self, X_batch, y_batch):
        """A training iteration step.

        Args:
            X_batch (torch.Tensor): A batch for input data
            y_batch (torch.Tensor): A batch of groundtruth

        Returns:
            tuple[torch.Tensor, float]:
                0: predictions of the model for input X_batch
                1: loss computed for batch
        """
        self._train_iter += 1

        # resets the gradients after every batch
        self.optimizer.zero_grad()

        X_batch, y_batch = X_batch.to(self.config.device), y_batch.to(
            self.config.device
        )
        predictions = self.model(X_batch)

        # compute loss and backpropage the gradients
        loss = self.criterion(predictions, y_batch)
        loss.backward()

        # update the weights
        self.optimizer.step()

        if self._train_iter % self.log_freq == 0:
            self.writer.add_scalar(
                "train/iteration/loss", loss.item(), self._train_iter
            )
            self.writer.add_scalar(
                "learning_rate", self.optimizer.param_groups[0]["lr"], self._train_iter
            )
            print(f"\nTraining - Loss at iteration {self._train_iter}: {loss.item()}", end=" ")

        if self.lr_scheduler:
            self.lr_scheduler.step()

        return predictions, loss

    def train_epoch(self):
        """Train the model for an epoch. Model weights are being updated ,and,
        loss of training is collected.

        Returns:
            float: The training loss for one epoch
        """
        assert self.train_loader is not None,\
            ValueError(
                "train_loader required to use epoch level training api of Learner \
                class."
            )
    
        # initialize every epoch
        epoch_loss = 0
        datset_size = len(self.train_loader.dataset)
        n_batches = len(self.train_loader)

        # set the model in training phase
        self.model.train()

        for batch_id, (X_batch, y_batch) in enumerate(self.train_loader):
            batch_size = len(X_batch)

            _, loss = self.train_batch(X_batch, y_batch)

            if self._train_iter % self.log_freq == 0:
                current = (batch_id + 1) * batch_size
                print(
                    f"| Epoch: {self._epoch} | "
                    + f"Progress: [{current}/{datset_size}]", end=" "
                )

            # loss
            epoch_loss += loss.item()

        print()
        return epoch_loss / n_batches

    def validate_batch(self, X_batch, y_batch):
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

        X_batch, y_batch = X_batch.to(self.config.device), y_batch.to(
            self.config.device
        )

        if self.model.training:  # Checks if the model is in training mode
            self.model.eval()

        # deactivates autograd
        with torch.no_grad():
            predictions = self.model(X_batch)

            # compute loss
            loss = self.criterion(predictions, y_batch)

        if self._valid_iter % self.log_freq == 0:
            self.writer.add_scalar(
                "validation/iteration/loss", loss.item(), self._valid_iter
            )
            print(f"\nValidation - Loss at iteration {self._valid_iter}: {loss.item()}", end=" ")

        return predictions, loss

    def validate_epoch(self):
        """Validate the model for an epoch on validation set. The validation
        loss is collected.

        Returns:
            tuple: A tuple of length 2 with validation loss and accuracy for one epoch
        """
        assert self.valid_loader is not None,\
            ValueError(
                "valid_loader required to use epoch level valiation api of Learner \
                 class."
            )


        # initialize every epoch
        epoch_loss = 0
        n_batches = len(self.valid_loader)
        datset_size = len(self.valid_loader.dataset)

        # deactivating dropout layers
        self.model.eval()

        for batch_id, (X_batch, y_batch) in enumerate(self.valid_loader):
            batch_size = len(X_batch)

            _, loss = self.validate_batch(X_batch, y_batch)

            # keep track of loss
            epoch_loss += loss.item()

            if self._valid_iter % self.log_freq == 0:
                current = (batch_id + 1) * batch_size
                print(
                    f"| Epoch: {self._epoch} | "
                    + f"Progress: [{current}/{datset_size}]", end=" "
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

    def save_checkpoint(self, result: dict):
        """To checkpoint model at any epoch of a trial by saving
        model.state_dict() and optimizer.state_dict().

        Args:
            path: Path to trial directory
        """

        os.makedirs(self.trial_dir, exist_ok=True)

        checkpoint_path = self.trial_dir / "checkpoint.pth"
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, str(checkpoint_path))

        config_path = self.trial_dir / "configuration.json"
        with open(config_path, "w") as config_file:
            config_file.write(self.config.model_dump_json(exclude={"device"}))

        result_path = self.trial_dir / "result.json"
        with open(result_path, "w") as result_file:
            result_file.write(result.model_dump_json())

    def load_checkpoint(self, path: str):
        """To load model checkpoint at any epoch of a trial by loading
        model.state_dict() and optimizer.state_dict().

        Args:
            path (str): A path for the trial checkpoint.
        """
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
