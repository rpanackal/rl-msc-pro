import os
from datetime import datetime
import json
import torch
from colorama import Fore, init
from ..core.datamodels import EpochResult

init(autoreset=True)

class Trainer:
    def __init__(
        self, train_loader, valid_loader, model, criterion, optimizer, config
    ) -> None:
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config

        self.incumbent_loss = float("inf")
        self.epoch = 0

        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.trial_name = f"{self.config.dataset.id}_{self.config.model.name}_{current_datetime}"
        config.name = self.trial_name

    def train_epoch(self):
        """Train the model for an epoch. Model weights are being updated ,and,
        loss of training is collected.

        Returns:
            float: The training loss for one epoch
        """
        # initialize every epoch
        epoch_loss = 0
        datset_size = len(self.train_loader.dataset)
        n_batches = len(self.train_loader)

        # set the model in training phase
        self.model.train()

        for batch_id, (X_batch, y_batch) in enumerate(self.train_loader):
            batch_size = len(X_batch)

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

            # loss
            epoch_loss += loss.item()

            if batch_id % 10 == 0:
                loss, current = loss.item(), (batch_id + 1) * batch_size
                print(
                    f"Training - Epoch: {self.epoch} | "
                    + f"Loss at iteration {batch_id}: {loss} | "
                    + f"Progress: [{current}/{datset_size}]"
                )

        return epoch_loss / n_batches

    def validate_epoch(self):
        """Validate the model for an epoch on validation set. The validation
        loss is collected.

        Returns:
            tuple: A tuple of length 2 with validation loss and accuracy for one epoch
        """

        # initialize every epoch
        epoch_loss = 0
        n_batches = len(self.valid_loader)
        datset_size = len(self.valid_loader.dataset)

        # deactivating dropout layers
        self.model.eval()

        # deactivates autograd
        with torch.no_grad():
            for batch_id, (X_batch, y_batch) in enumerate(self.valid_loader):
                batch_size = len(X_batch)

                X_batch, y_batch = X_batch.to(self.config.device), y_batch.to(
                    self.config.device
                )
                predictions = self.model(X_batch)

                # compute loss
                loss = self.criterion(predictions, y_batch)

                # keep track of loss
                epoch_loss += loss.item()

                if batch_id % 10 == 0:
                    loss, current = loss.item(), (batch_id + 1) * batch_size
                    print(
                        f"Validation - Epoch: {self.epoch} | "
                        + f"Loss at iteration {batch_id}: {loss} | " 
                        + f"Progress: [{current}/{datset_size}]"
                    )

        return epoch_loss / n_batches

    def step(
        self,
    ):
        """
        One logical iteration step for training, here an epoch.

        Returns:
            dict: A dictionary of metrics to be used for comparing
        """
        self.epoch += 1

        # train the model
        train_loss = self.train_epoch()

        # validate the model
        valid_loss = self.validate_epoch()

        incumbent_found = valid_loss < self.incumbent_loss
        if incumbent_found:
            self.incumbent_loss = valid_loss

        print(
            Fore.GREEN
            + f"\nEpoch: {self.epoch} complete | "
            + f"Training Loss: {train_loss} | "
            + f"Validation Loss: {valid_loss} | "
            + f"Incumbent Loss: {self.incumbent_loss}\n"
        )

        return EpochResult(**{
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "epoch": self.epoch,
            "incumbent_loss": self.incumbent_loss,
            "incumbent_found": incumbent_found 
        })

    def save_checkpoint(self, result: dict):
        """To checkpoint model at any epoch of a trial by saving
        model.state_dict() and optimizer.state_dict().

        Args:
            path: Path to trial directory
        """
        
        trail_dir = self.config.checkpoint_dir / self.trial_name
        os.makedirs(trail_dir, exist_ok=True)

        checkpoint_path = trail_dir / "checkpoint.pth"
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }   
        torch.save(checkpoint, str(checkpoint_path))

        config_path = trail_dir / "configuration.json"
        with open(config_path, "w") as config_file:
            config_file.write(self.config.model_dump_json(exclude={"device"}))

        result_path = trail_dir / "result.json"
        with open(result_path, "w") as result_file:
            result_file.write(result.model_dump_json())

    def load_checkpoint(self, path):
        """To load model checkpoint at any epoch of a trial by loading
        model.state_dict() and optimizer.state_dict().

        Args:
            path (str): A directory for the trail checkpoint.
        """
        model_state, optimizer_state = torch.load(path)

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)