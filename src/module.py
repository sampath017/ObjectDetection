import torch.nn.functional as F
import torch
from torch import nn
from utils import accuracy


class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128, max_pool=True),
            ResidualBlock(128),

            ConvBlock(128, 256, max_pool=True),
            ConvBlock(256, 512, max_pool=True),
            ResidualBlock(512),
            nn.MaxPool2d(kernel_size=4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        return logits


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=64,
        max_pool=False,
        last_block=False
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=True if last_block else False
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        ]

        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels=3,
        last_block=False,
    ):
        super().__init__()
        self.block = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),

            # Block 2
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=True if last_block else False
            ),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x) + x

        return x


class QuickModule:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, batch):
        self.model = self.model.to(self.device)
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)

        return loss, acc


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*32*32, 10),
        )

    def forward(self, x):
        logits = self.classifier(x)

        return logits


class ResidualNetModule(QuickModule):
    def __init__(self, toy_model=False):
        super().__init__()
        self.model = ToyNet() if toy_model else ResidualNet()

    def training_step(self, batch):
        loss, acc = self.forward(batch)

        return loss, acc

    def validation_step(self, batch):
        loss, acc = self.forward(batch)

        return loss, acc
