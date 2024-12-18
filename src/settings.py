project_name = "ImageClassification"

model = {
    "name": "ResNet9",
    "num_layers": 9
}

dataset = {
    "name": "CIFAR100",
    "batch_size": 256,
    "train_split": 0.7,
    "val_split": 0.3,
    "augumentations": True
}

max_epochs = 50

optimizer = {
    "name": "Adam",
    "weight_decay": None
}

# lr_scheduler = {
#     "name": "OneCycleLR",
#     "max_lr": 0.01,
# }

lr_scheduler = None

wandb_offline = False
fast_dev_run = False
