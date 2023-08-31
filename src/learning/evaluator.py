import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from ..core.datamodels import EvaluationEpochResult
from datetime import datetime
from colorama import Fore, init
from pathlib import PurePath
from typing import Callable


init(autoreset=True)


class SupervisedEvaluator:
    """A class that handles testing of a learnt model."""

    def __init__(
        self,
        test_loader: DataLoader,
        model: nn.Module,
        criterion,
        config,
        custom_to_model: Callable | None = None,
        custom_to_criterion: Callable | None = None,
        writer: SummaryWriter | None = None,
        log_freq: int = 1,
        checkpoint_dir: str | None = None,
    ) -> None:
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.config = config
        self.device = self.config.device

        self.custom_to_model = custom_to_model
        self.custom_to_criterion = custom_to_criterion

        self.log_freq = log_freq
        self.trial_dir = checkpoint_dir
        self.writer = writer

        # When trial_dir not set
        if not self.trial_dir and self.writer:
            self.trial_dir = PurePath(writer.get_logdir())

        path = self.trial_dir / "checkpoint.pth"
        self.load_checkpoint(str(path))

        self._test_iter = -1

    def test(self):
        """
        One logical iteration step for testing, here an epoch.

        Returns:
            dict: A dictionary of metrics to be used for comparing
        """
        test_loss = 0
        n_batches = len(self.test_loader)
        datset_size = len(self.test_loader.dataset)

        if self.model.training:  # Checks if the model is in training mode
            self.model.eval()

        for batch_id, batch in enumerate(self.test_loader):
            batch_size = self.test_loader.batch_size

            self._test_iter += 1

            args, kwargs = self.to_model(batch)

            with torch.no_grad():
                output = self.model(*args, **kwargs)

                # Compute loss
                args, kwargs = self.to_criterion(batch, output)
                loss = self.criterion(*args, **kwargs)

            if batch_id % self.log_freq == 0:
                loss, current = loss.item(), (batch_id + 1) * batch_size
                self.writer.add_scalar("test/iteration/loss", loss, self._test_iter)
                print(
                    f"Testing - "
                    + f"Loss at iteration {batch_id}: {loss} | "
                    + f"Progress: [{current}/{datset_size}]"
                )

            test_loss += loss

        test_loss /= n_batches

        print(Fore.GREEN + f"Final average testing Loss: {test_loss}")

        return EvaluationEpochResult(loss=test_loss)

    def load_checkpoint(self, path: str):
        """To load model checkpoint of a trial by loading
        model.state_dict().

        Args:
            path (str): A directory for the trial checkpoint.
        """
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint["model_state_dict"])

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
        X_batch, y_batch = batch
        
        args = [X_batch.to(self.device), y_batch.to(self.device)]
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