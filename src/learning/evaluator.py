import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from ..core.datamodels import EvaluationEpochResult
from datetime import datetime
from colorama import Fore, init
from pathlib import PurePath


init(autoreset=True)

class SupervisedEvaluator():
    """A class that handles testing of a learnt model.
    """
    def __init__(
        self,
        test_loader: DataLoader,
        model: nn.Module,
        criterion,
        config,
        writer: SummaryWriter | None = None,
        log_freq: int = 1,
        trial_dir: str | None = None

    ) -> None:
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.config = config

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
                trial_name = f"{self.config.dataset.id}_{self.config.model.name}_{current_datetime}"
                self.trial_dir = self.config.checkpoint_dir / trial_name

        self.writer = writer if writer else SummaryWriter(log_dir=str(self.trial_dir))

        path = self.trial_dir / "checkpoint.pth"
        self.load_checkpoint(str(path))
        
        self.test_iter = -1

    def test(self):
        """
        One logical iteration step for testing, here an epoch.

        Returns:
            dict: A dictionary of metrics to be used for comparing
        """
        test_loss = 0
        n_batches = len(self.test_loader)
        datset_size = len(self.test_loader.dataset)

        self.model.eval()
        with torch.no_grad():
            for batch_id, (X_batch, y_batch) in enumerate(self.test_loader):
                batch_size = len(X_batch)
                self.test_iter += 1

                X_batch, y_batch = X_batch.to(self.config.device), y_batch.to(self.config.device)

                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)

                if batch_id % self.log_freq == 0:
                    loss, current = loss.item(), (batch_id + 1) * batch_size
                    self.writer.add_scalar("test/iteration/loss", loss, self.test_iter)
                    print(
                        f"Testing - "
                        + f"Loss at iteration {batch_id}: {loss} | "
                        + f"Progress: [{current}/{datset_size}]"
                    )

                test_loss += loss
    
        test_loss /= n_batches

        print(
            Fore.GREEN
            + f"Final average testing Loss: {test_loss}"
        )
        
        return EvaluationEpochResult(loss=test_loss)


    def load_checkpoint(self, path: str):
        """To load model checkpoint of a trial by loading
        model.state_dict().

        Args:
            path (str): A directory for the trial checkpoint.
        """
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint["model_state_dict"])