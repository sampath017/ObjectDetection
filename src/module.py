import torch.nn.functional as F
from utils import accuracy
from models import ToyNet, ResNet18, ResNet9, ResNet56
from quickai.module import QuickModule


class ResNetModule(QuickModule):
    def __init__(self, model=None, toy_model=False):
        super().__init__()
        if model:
            self.model = model
        elif toy_model:
            self.model = ToyNet()
        else:
            self.model = ResNet18(num_classes=100)

    def training_step(self, batch):
        loss, acc = self.forward(batch)

        return loss, acc

    def validation_step(self, batch):
        loss, acc = self.forward(batch)

        return loss, acc
