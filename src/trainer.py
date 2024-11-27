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
        optimizer=None,
        callbacks=[],
        logs_path=None,
        lr_scheduler=None,
        device=None,
        limit_train_batches=None,
        limit_val_batches=None
    ):
        self.module = module
        self.module.device = device
        self.module.logger = logger

        self.logger = logger
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.logs_path = logs_path

    def overfit_callback(self):
        for callback in self.callbacks:
            if isinstance(callback, OverfitCallback):
                return callback

    def fit(self, train_dataloader, val_dataloader):
        self.logger.init()
        epochs = self.logger.config["epochs"]

        # Overfit callback
        train_dataloader = DataLoader(
            train_dataloader.dataset, batch_size=train_dataloader.batch_size, shuffle=False)
        callback = self.overfit_callback()
        if callback:
            self.limit_train_batches = callback.limit_train_batches
            self.limit_val_batches = callback.limit_val_batches

        # Loop
        for epoch in range(epochs):
            self.epoch = epoch
            epoch_train_accuracy = self.train(train_dataloader)
            epoch_val_accuracy = self.val(val_dataloader)

            self.log_model()
            # fmt:off
            print(f"Epoch: {self.epoch}, train_accuracy: {\
                  epoch_train_accuracy:.2f}, val_accuracy: {epoch_val_accuracy:.2f}")
            # fmt:on

    def train(self, train_dataloader):
        step_train_losses = []
        step_train_accuracies = []

        self.module.model.train()
        for step, batch in enumerate(train_dataloader):
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
            "epoch_val_loss": epoch_val_loss,
            "epoch_val_accuracy": epoch_val_accuracy,
        })

        return epoch_val_accuracy

    # @torch.no_grad()
    # def test(self, test_dataloader):
    #     step_test_losses = []
    #     step_test_accuracies = []

    #     self.model.eval()
    #     for batch in test_dataloader:
    #         loss, acc = self._forward(batch)
    #         step_test_losses.append(loss.item())
    #         step_test_accuracies.append(acc.item())

    #     test_loss = torch.tensor(step_test_losses).mean().item()
    #     test_accuracy = torch.tensor(step_test_accuracies).mean().item()

    #     return test_loss, test_accuracy

    def load_model(self, run_path, model_path):
        api = wandb.Api()
        run = api.run(run_path)
        self.config = run.config

        artifact = api.artifact(model_path)
        local_path = artifact.download()
        file_name = artifact.file().split("/")[-1]

        self.model = VGGNet()
        self.model.load_state_dict(torch.load(
            Path(local_path)/file_name, map_location=self.device))

    def log_model(self):
        model_path = self.logs_path / f"model_{self.epoch}.pt"
        torch.save(self.model.state_dict(), model_path)
        wandb.log_model(model_path, aliases=[f"[epoch-{self.epoch}]"])
