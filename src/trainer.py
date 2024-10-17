import torch
import wandb
import torch.nn.functional as F
from module import ResNet
from pathlib import Path


class Trainer:
    def __init__(self, model=None, config=None, optimizer=None, lr_scheduler=None, device=None, limit_batches=2):
        if model:
            self.model = model.to(device)

        self.config = config
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.limit_batches = limit_batches

    def fit(self, train_dataloader, val_dataloader):
        epochs = self.config["epochs"]
        for epoch in range(epochs):
            self.epoch = epoch
            epoch_train_accuracy = self.train(train_dataloader)
            epoch_val_accuracy = self.val(val_dataloader)

            self.log_model()
            # fmt:off
            print(f"Epoch: {self.epoch}, train_accuracy: {\
                  epoch_train_accuracy:.2f}, val_accuracy: {epoch_val_accuracy:.2f}")
            # fmt:on

    def _forward(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        accuracy = self._accuracy(logits, y)

        return loss, accuracy

    @torch.no_grad()
    def _accuracy(self, logits, y):
        probs = F.softmax(logits, dim=-1)
        y_pred = probs.argmax(dim=-1)
        accuracy = 100 * ((y_pred == y).sum() / y_pred.shape[0])

        return accuracy

    def train(self, train_dataloader):
        step_train_losses = []
        step_train_accuracies = []
        self.model.train()
        for i, batch in enumerate(train_dataloader):
            if (self.limit_batches is not None):
                if (i > self.limit_batches):
                    break

            loss, accuracy = self._forward(batch)
            step_train_losses.append(loss.item())
            step_train_accuracies.append(accuracy.item())
            wandb.log({
                "lr": self.lr_scheduler.get_last_lr()[0],
                "train_loss": loss.item(),
                "train_accuracy": accuracy.item(),
            })

            # optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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
        self.model.eval()
        for i, batch in enumerate(val_dataloader):
            if (self.limit_batches is not None):
                if (i > self.limit_batches):
                    break

            loss, accuracy = self._forward(batch)
            step_val_losses.append(loss.item())
            step_val_accuracies.append(accuracy.item())
            wandb.log({
                "step_val_loss": loss.item(),
                "step_val_accuracy": accuracy.item(),
            })

        epoch_val_loss = torch.tensor(step_val_losses).mean()
        epoch_val_accuracy = torch.tensor(step_val_accuracies).mean()

        wandb.log({
            "epoch": self.epoch,
            "epoch_val_loss": epoch_val_loss,
            "epoch_val_accuracy": epoch_val_accuracy,
        })

        return epoch_val_accuracy

    @torch.no_grad()
    def test(self, test_dataloader):
        step_test_losses = []
        step_test_accuracies = []
        self.model.eval()
        for batch in test_dataloader:
            loss, accuracy = self._forward(batch)
            step_test_losses.append(loss.item())
            step_test_accuracies.append(accuracy.item())

        test_loss = torch.tensor(step_test_losses).mean().item()
        test_accuracy = torch.tensor(step_test_accuracies).mean().item()

        return test_loss, test_accuracy

    def load_model(self, run_path, model_path):
        api = wandb.Api()
        run = api.run(run_path)
        self.config = run.config

        artifact = api.artifact(model_path)
        local_path = artifact.download()
        file_name = artifact.file().split("/")[-1]

        self.model = ResNet()
        self.model.load_state_dict(torch.load(
            Path(local_path)/file_name, map_location=self.device))

    def log_model(self):
        model_path = f"model_{self.epoch}.pt"
        torch.save(self.model.state_dict(), model_path)
        wandb.log_model(model_path, aliases=[f"[epoch-{self.epoch}]"])
