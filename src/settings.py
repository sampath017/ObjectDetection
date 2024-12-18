project_name = "ImageClassification"

model = {
    "name": "ResNet18",
    "num_layers": 18
}

dataset = {
    "name": "CIFAR100",
    "batch_size": 256,
    "train_split": 0.7,
    "val_split": 0.3
}

max_epochs = 20

optimizer = {
    "name": "Adam",
    "weight_decay": None
}

lr_scheduler = {
    "name": "OneCycleLR",
    "max_lr": 0.01,
}

wandb_offline = True
fast_dev_run = True
