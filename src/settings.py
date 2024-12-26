project_name = "ImageClassification"

model = {
    "name": "ResNet18",
    "num_layers": 18,
}

dataset = {
    "name": "CIFAR100",
    "batch_size": 512,
    "train_split": 0.7,
    "val_split": 0.3
}

max_epochs = 50

optimizer = {
    "name": "Adam",
    "weight_decay": None
}

# lr_scheduler = None
lr_scheduler = {
    "name": "OneCycleLR",
    "max_lr": 0.01,
}

test_run = False
limit_batches = None
wandb_offline = False
fast_dev_run = False
