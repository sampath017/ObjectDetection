project_name = "ImageClassification"

model = {
    "name": "ResNet56",
    "num_layers": 56
}

dataset = {
    "name": "CIFAR100",
    "batch_size": 32,
    "train_split": 0.7,
    "val_split": 0.3,
    "augumentations": True
}

max_epochs = 30

optimizer = {
    "name": "Adam",
    "weight_decay": None
}

# lr_scheduler = "default"

lr_scheduler = {
    "name": "OneCycleLR",
    "max_lr": 0.01,
}

wandb_offline = False
fast_dev_run = False
