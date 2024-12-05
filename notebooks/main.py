import torch
from torchsummary import summary
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from pathlib import Path
import wandb

from trainer import Trainer
from module import VGGNetModule
from utils import model_size
from callbacks import OverfitCallback, EarlyStoppingCallback
from logger import WandbLogger

data_path = Path("../data")
logs_path = Path("../logs")
logs_path.mkdir(exist_ok=True)

logger = WandbLogger(
    project_name="ImageClassification",
    config={
        "model_architecture": "VGGNet",
        "batch_size": 1024,
        "max_epochs": 100,
        "optimizer": {
            "name": "Adam",
        },
        "train_split": 42_000,
        "val_split": 8000
    },
    logs_path=logs_path
)

dataset = CIFAR10(data_path, train=True, download=True, transform=v2.Compose([
    # Normalize
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
]))

train_dataset, val_dataset = random_split(
    dataset, [logger.config["train_split"], logger.config["val_split"]])

train_dataloader = DataLoader(
    train_dataset, batch_size=logger.config["batch_size"], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=logger.config["batch_size"])

# %%
callbacks = [
    EarlyStoppingCallback(min_val_accuracy=80.0, accuracy_diff=4.0, wait_epochs=5),
    OverfitCallback(limit_batches=2, max_epochs=200),
]

# %%
module = VGGNetModule()

trainer = Trainer(
    module=module,
    logger=logger,
    callbacks=callbacks,
    logs_path=logs_path,
    fast_dev_run=False,
    measure_time=True
)

try:
    trainer.fit(train_dataloader, val_dataloader)
except KeyboardInterrupt as e:
    print("Run stopped!")
finally:
    wandb.finish()
