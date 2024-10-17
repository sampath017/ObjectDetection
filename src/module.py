from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=num_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=num_channels),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.res_block(x) + x

        return logits


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),

            ResidualBlock(128),

            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),

            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),

            ResidualBlock(512),

            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), padding=(0, 0)),
        )

        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        features = self.feature_extractor(x).flatten(start_dim=1)
        logits = self.classifier(features)

        return logits
