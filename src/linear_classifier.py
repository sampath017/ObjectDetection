import torch
import torch.nn.functional as F

from torch import nn
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, optimizer, logger, device, limit_batches=2, reg=False):
        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.device = device
        self.limit_batches = limit_batches
        self.reg = reg

    def fit(self, train_dataloader, val_dataloader):
        epochs = self.logger.config["epochs"]
        for epoch in range(epochs):
            self.train(train_dataloader, epoch, self.limit_batches)
            self.val(val_dataloader, epoch, self.limit_batches)

            self.logger.log_model(self.model, epoch)
            print(f"Epoch: {epoch}")

    def _forward(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        self.model = self.model.to(self.device)

        logits = self.model(x)
        forward_loss = F.cross_entropy(logits, y)
        reg_loss = 0.0
        if self.reg:
            reg_loss = torch.tensor([p.square().sum()
                                    for p in self.model.parameters()]).sum()

        loss = forward_loss + reg_loss
        acc = self._accuracy(logits, y)

        return loss, acc

    @torch.no_grad()
    def _accuracy(self, logits, y):
        probs = F.softmax(logits, dim=-1)
        y_pred = probs.argmax(dim=-1)
        accuracy = 100 * ((y_pred == y).sum() / y_pred.shape[0])

        return accuracy

    def train(self, train_dataloader, epoch):
        for i, batch in enumerate(train_dataloader):
            if (self.limit_batches is not None):
                if (i > self.limit_batches):
                    break

            loss, acc = self._forward(batch)
            self.logger.log_metric(loss.item(), "train_loss", epoch)
            self.logger.log_metric(acc.item(), "train_accuracy", epoch)

            # optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def val(self, val_dataloader, epoch):

        for i, batch in enumerate(val_dataloader):
            if (self.limit_batches is not None):
                if (i > self.limit_batches):
                    break

            loss, acc = self._forward(batch)
            self.logger.log_metric(loss.item(), "val_loss", epoch)
            self.logger.log_metric(acc.item(), "val_accuracy", epoch)


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3*32*32, 50),
            nn.ReLU(),

            nn.Linear(50, 100),
            nn.ReLU(),

            nn.Linear(100, 50),
            nn.ReLU(),

            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        logits = self.model(x)

        return logits


class TempModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3*32*32, 10),
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        logits = self.model(x)

        return logits
