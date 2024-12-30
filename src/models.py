import torch
from torch import nn


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ResBlock(128),

            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 512),
            ResBlock(512),

            ConvBlock(512, 512, stride=2),
            ConvBlock(512, 512),
            ResBlock(512),

            ConvBlock(512, 512, stride=2),
            ConvBlock(512, 512),
            ResBlock(512),

            ConvBlock(512, 512, last_block=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes),
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
        stride=1,
        max_pool=False,
        last_block=False
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=1,
                stride=stride,
                bias=True if last_block else False
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        ]

        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        channels=3,
    ):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels)
        )

    def forward(self, x):
        x = self.block(x) + x

        return x


class ToyNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.num_classes = num_classes
        self.param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        logits = self.param * \
            torch.empty((x.shape[0], self.num_classes), dtype=torch.float)

        return logits
