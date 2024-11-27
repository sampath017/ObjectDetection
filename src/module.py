import torch.nn.functional as F
import torch
from torch import nn
from utils import accuracy


class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),

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

            # Block 2
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
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 10),
        )

    def forward(self, x):
        features = self.feature_extractor(x).flatten(start_dim=1)
        logits = self.classifier(features)

        return logits


class QuickModule:
    def __init__(self):
        pass

    def forward(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)

        return loss, acc


class VGGNetModule(QuickModule):
    def __init__(self):
        self.model = VGGNet()

    def training_step(self, batch):
        loss, acc = self.forward(batch)

        return loss, acc

    def validation_step(self, batch):
        loss, acc = self.forward(batch)

        return loss, acc

    def optimizer(self):
        return torch.optim.Adam(params=self.model.parameters())
