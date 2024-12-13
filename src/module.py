import torch.nn.functional as F
import torch
from torch import nn
from utils import accuracy


class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 8, 8, 16, max_pool=True),
            ConvBlock(16, 32, 32, 64, max_pool=True),
            ConvBlock(64, 128, 128, 512, max_pool=True),
            ResidualBlock(512, max_pool=True),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            # ResidualBlock(512, 512, 512, 512),

            # ResidualBlock(512, 512, 512, 512),
            # ResidualBlock(512, 512, 512, 512),
            # ResidualBlock(512, 512, 512, 512),
            # ResidualBlock(512, 512, 512, 512),
            # ResidualBlock(512, 512, 512, 512),
            # ResidualBlock(512, 512, 512, 512),
            # ResidualBlock(512, 512, 512, 512),
            # ResidualBlock(512, 512, 512, 512),
            # ResidualBlock(512, 512, 512, 512),
            # ResidualBlock(512, 512, 512, 512),

            # ResidualBlock(512, 512, 512, 512),
            # ResidualBlock(512, 512, 512, 512),
            # ResidualBlock(512, 512, 512, 512),
            # ResidualBlock(512, 512, 512, 512),
            # ResidualBlock(512, 512, 512, 512),
            # ResidualBlock(512, 512, 512, 512),
            # ResidualBlock(512, 512, 512, 512),
            ResidualBlock(512, last_block=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),

            nn.Linear(128, 10),
        )

    def forward(self, x):
        features = self.feature_extractor(x).flatten(start_dim=1)
        logits = self.classifier(features)

        return logits


class ConvBlock(nn.Module):
    def __init__(
        self,
        block1_in_channels=3,
        block1_out_channels=8,
        block2_in_channels=8,
        block2_out_channels=8,
        max_pool=False,
        last_block=False
    ):
        super().__init__()
        layers = [
            # Block 1
            nn.Conv2d(
                in_channels=block1_in_channels,
                out_channels=block1_out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=block1_out_channels),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=block2_in_channels,
                out_channels=block2_out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=True if last_block else False
            )
        ]

        if not last_block:
            layers.extend([nn.BatchNorm2d(num_features=block2_out_channels),
                           nn.ReLU(),])

        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels=3,
        max_pool=False,
        last_block=False,
    ):
        super().__init__()
        layers = [
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

            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=True if last_block else False
            )
        ]

        if not last_block:
            layers.extend([nn.BatchNorm2d(num_features=channels),
                           nn.ReLU(),])
        self.block = nn.Sequential(*layers)

        self.max_pool = nn.MaxPool2d(kernel_size=(
            2, 2), stride=2) if max_pool else None

    def forward(self, x):
        x = self.block(x) + x

        if self.max_pool:
            x = self.max_pool(x)

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

    def load_from_checkpoint(self, path):
        self.optimizer_func = self.optimizer()
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer_func.load_state_dict(checkpoint["optimizer"])


class ResidualNetModule(QuickModule):
    def __init__(self):
        super().__init__()
        self.model = ResidualNet()

    def training_step(self, batch):
        loss, acc = self.forward(batch)

        return loss, acc

    def validation_step(self, batch):
        loss, acc = self.forward(batch)

        return loss, acc

    def optimizer(self):
        return torch.optim.Adam(params=self.model.parameters())
