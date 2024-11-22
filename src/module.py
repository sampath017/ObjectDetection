import torch.nn.functional as F
from torch import nn


class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, 100),
        )

    def forward(self, x):
        features = self.feature_extractor(x).flatten(start_dim=1)
        logits = self.classifier(features)

        return logits


class QuickModule:
    def __init__(self):
        pass

    def log(self, metric, metric_name):
        self.logger.log(metric_name, metric)


class VGGNetModule(QuickModule):
    def __init__(self, model):
        self.model = model

    def forward(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        accuracy = self._accuracy(logits, y)

        return loss, accuracy

    def training_step(self, batch):
        loss, acc = self.forward(batch)

        self.log("train_loss", loss)
        self.log("train_accuracy", acc)

        return loss

    def validation_step(self, batch):

        loss, acc = self.forward(batch)

        self.log("val_loss", loss)
        self.log("val_accuracy", acc)

        return loss
