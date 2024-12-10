import os
import torch
import wandb
from torch.utils.data import DataLoader
from pathlib import Path
from callbacks import OverfitCallback, EarlyStoppingCallback
import time
from dataset import MapDataset


class Trainer:
    def __init__(
        self,
        module=None,
        logger=None,
        callbacks=[],
        logs_path=None,
        lr_scheduler=None,
        limit_train_batches=None,
        limit_val_batches=None,
        fast_dev_run=False,
        measure_time=False,
        num_workers=6,
        checkpoint="every"
    ):
        # module
        self.module = module
        self.module.logger = logger

        self.logger = logger
        self.logs_path = logs_path
        self.optimizer_func = module.optimizer()
        self.callbacks = callbacks
        self.lr_scheduler = lr_scheduler
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.fast_dev_run = fast_dev_run
        self.measure_time = measure_time
        self.num_workers = num_workers
        self.checkpoint = checkpoint
        self.overfit_callback = None

        if self.fast_dev_run:
            os.environ["WANDB_MODE"] = "offline"
        else:
            os.environ["WANDB_MODE"] = "online"

    def setup(self):
        self.logger.init()
        wandb.log({"model_architecture": self.module.model})
        self.max_epochs = self.logger.config["max_epochs"]

        # Fast dev run
        if self.fast_dev_run:
            self.limit_train_batches = 5
            self.limit_val_batches = 5
            self.max_epochs = 1
        # Overfit callback
        else:
            self.overfit_callback = self._overfit_callback()
            if self.overfit_callback:
                if self.overfit_callback.augument_data:
                    train_dataset = self.train_dataloader.dataset
                else:
                    train_dataset = MapDataset(
                        self.train_dataloader.dataset, transform=self.val_dataloader.dataset.transform)

                self.train_dataloader = DataLoader(
                    train_dataset, batch_size=self.overfit_callback.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
                self.limit_train_batches = self.overfit_callback.limit_train_batches
                self.limit_val_batches = self.overfit_callback.limit_val_batches
                self.max_epochs = self.overfit_callback.max_epochs

    def _overfit_callback(self):
        for callback in self.callbacks:
            if isinstance(callback, OverfitCallback):
                return callback

    def _earlystopping_callback(self, epoch_train_accuracy, epoch_val_accuracy):
        stop_training = False
        accuracy_diff = None
        for callback in self.callbacks:
            if isinstance(callback, EarlyStoppingCallback):
                stop_training, accuracy_diff = callback.check(
                    epoch_train_accuracy, epoch_val_accuracy)
                break

        return stop_training, accuracy_diff

    def fit(self, train_dataloader, val_dataloader):
        # setup
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.setup()
        best_epoch_train_loss = torch.inf

        try:
            # Loop
            for epoch in range(self.max_epochs):
                measure_time_bool = self.measure_time and epoch == 0
                self.epoch = epoch
                if measure_time_bool:
                    start_time = time.time()
                epoch_train_loss, epoch_train_accuracy = self.train()
                if measure_time_bool:
                    end_time = time.time()

                if measure_time_bool:
                    # type: ignore
                    print(f"Time per epoch: {end_time-start_time:.2f} seconds")

                if not self.overfit_callback:
                    epoch_val_loss, epoch_val_accuracy = self.val(
                        val_dataloader)
                else:
                    epoch_val_accuracy = torch.inf

                # Print
                # fmt:off
                print(f"Epoch: {self.epoch}, train_accuracy: {\
                    epoch_train_accuracy:.2f}, val_accuracy: {epoch_val_accuracy:.2f}")

                if self.fast_dev_run:
                    print("Sanity check done with fast dev run!")

                if hasattr(self, 'overfit_callback') and self.overfit_callback and epoch_train_accuracy >= 100.0:
                    print(f"Overfit done at epoch: {epoch}.")
                    break
                # fmt:on

                if self.checkpoint == "every":
                    self.save_checkpoint()
                if self.checkpoint == "best_train":
                    if epoch_train_loss < best_epoch_train_loss:
                        best_epoch_train_loss = epoch_train_loss
                        self.save_checkpoint(save_best_model=True)

                # Early Stopping
                stop_training, accuracy_diff = self._earlystopping_callback(
                    epoch_train_accuracy, epoch_val_accuracy)
                if stop_training:
                    # fmt:off
                    print(f"Stoppping training due to early stopping crossing threshold {\
                        accuracy_diff:.2f}")
                    # fmt:on
                    break

            if epoch == self.max_epochs:
                print(f"Training stopped max_epochs: {
                      self.max_epochs} reached!")

        except Exception as e:
            print(e)
        finally:
            if not self.fast_dev_run and self.save_best_model:
                wandb.log_model(self.save_path, aliases=["best"])

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
            self.optimizer_func.zero_grad()
            loss.backward()
            self.optimizer_func.step()

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

        return epoch_train_loss, epoch_train_accuracy

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

        return epoch_val_loss, epoch_val_accuracy

    def save_checkpoint(self, save_best_model=False):
        self.save_best_model = save_best_model
        self.checkpoint_path = Path(wandb.run.dir).parent / "checkpoints"
        self.checkpoint_path.mkdir(exist_ok=True)
        if save_best_model:
            self.save_path = self.checkpoint_path / f"best.pt"
        else:
            self.save_path = self.checkpoint_path / \
                f"checkpoint-{self.epoch}.pt"

        torch.save({
            "model": self.module.model.state_dict(),
            "optimizer": self.optimizer_func.state_dict()

        }, self.save_path)

        if not save_best_model:
            wandb.log_model(self.save_path, aliases=[
                            f"[checkpoint-{self.epoch}]"])
