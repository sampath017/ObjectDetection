project_name = "ImageClassification"

model = {
    "name": "ResNet18",
    "num_layers": 18,
}

dataset = {
    "name": "CIFAR10",
    "batch_size": 512,
    "train_split": 0.7,
    "val_split": 0.3
}

max_epochs = 50

optimizer = {
    "name": "AdamW",
    "weight_decay": 0.01
}

# lr_scheduler = None
lr_scheduler = {
    "name": "OneCycleLR",
    "max_lr": 0.01,
}

# transfer_learning = None
transfer_learning = {
    "freeze_fe": True,
    "change_fc": True
}

limit_batches = None
test_run = False
wandb_offline = False
fast_dev_run = False
