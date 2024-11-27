import torch
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from module import VGGNet
from pathlib import Path
from callbacks import OverfitCallback


class Trainer:
    def __init__(
        self,
        module=None,
        logger=None,
        callbacks=[],
        logs_path=None,
        lr_scheduler=None,
        device=None,
        limit_train_batches=None,
        limit_val_batches=None,
        fast_dev_run=False
    ):
        self.module = module
        self.module.device = device
        self.module.logger = logger

        self.logger = logger
        self.optimizer = module.optimizer()
        self.callbacks = callbacks
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.logs_path = logs_path
        self.fast_dev_run = fast_dev_run

    def setup(self):
        # Fast dev run
        if self.fast_dev_run:
            self.limit_train_batches = 5
            self.limit_val_batches = 5
            self.epochs = 1
        # Overfit callback
        else:
            self.overfit_callback = self._overfit_callback()
            if self.overfit_callback:
                self.train_dataloader = DataLoader(
                    self.train_dataloader.dataset, batch_size=self.train_dataloader.batch_size, shuffle=False)
                self.limit_train_batches = self.overfit_callback.limit_train_batches
                self.limit_val_batches = self.overfit_callback.limit_val_batches
                self.epochs = self.overfit_callback.max_epochs

    def _overfit_callback(self):
        for callback in self.callbacks:
            if isinstance(callback, OverfitCallback):
                return callback

    def fit(self, train_dataloader, val_dataloader):
        self.logger.init()
        self.epochs = self.logger.config["epochs"]

        # setup
        self.train_dataloader = train_dataloader
        self.setup()

        # Loop
        for epoch in range(self.epochs):
            self.epoch = epoch
            epoch_train_accuracy = self.train()
            epoch_val_accuracy = self.val(val_dataloader)

            self.log_model()
            # fmt:off
            print(f"Epoch: {self.epoch}, train_accuracy: {\
                  epoch_train_accuracy:.2f}, val_accuracy: {epoch_val_accuracy:.2f}")

            if self.fast_dev_run:
                print("Sanity check done with fast dev run!")

            if hasattr(self, 'overfit_callback') and self.overfit_callback and epoch_train_accuracy >= 100.0:
                print(f"Overfit done at epoch: {epoch}.")
                break
            # fmt:on

    def train(self):
        step_train_losses = []
        step_train_accuracies = []

        self.module.model.train()
        for step, batch in enumerate(self.train_dataloader):
            if self.limit_train_batches and (step > self.limit_train_batches):
                break

            loss, acc = self.module.training_step(batch)
            step_train_losses.append(loss.item())
            step_train_accuracies.append(acc.item())
            wandb.log({
                "step": step,
            })

            # optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # lr_scheduler step
            if self.lr_scheduler:
                self.lr_scheduler.step()

        epoch_train_loss = torch.tensor(step_train_losses).mean()
        epoch_train_accuracy = torch.tensor(step_train_accuracies).mean()

        wandb.log({
            "epoch": self.epoch,
            "epoch_train_loss": epoch_train_loss,
            "epoch_train_accuracy": epoch_train_accuracy,
        })

        return epoch_train_accuracy

    @torch.no_grad()
    def val(self, val_dataloader):
        step_val_losses = []
        step_val_accuracies = []

        self.module.model.eval()
        for step, batch in enumerate(val_dataloader):
            if self.limit_val_batches and (step > self.limit_val_batches):
                break

            loss, acc = self.module.validation_step(batch)
            step_val_losses.append(loss.item())
            step_val_accuracies.append(acc.item())

        epoch_val_loss = torch.tensor(step_val_losses).mean()
        epoch_val_accuracy = torch.tensor(step_val_accuracies).mean()

        wandb.log({
            "epoch": self.epoch,
            "epoch_val_loss": epoch_val_loss,
            "epoch_val_accuracy": epoch_val_accuracy,
        })

        return epoch_val_accuracy

    def log_model(self):
        model_path = self.logs_path / f"model_{self.epoch}.pt"
        torch.save(self.module.model.state_dict(), model_path)
        wandb.log_model(model_path, aliases=[f"[epoch-{self.epoch}]"])
