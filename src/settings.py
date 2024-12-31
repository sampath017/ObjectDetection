project_name = "ObjectDetection"

model = {
    "name": "ResNet18",
    "num_layers": 18,
}

dataset = {
    "name": "VOCDetection",
    "batch_size": 32,
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

transfer_learning = None
# transfer_learning = {
#     "freeze_fe": False,
#     "change_fc": False
# }

limit_batches = None
test_run = False
wandb_offline = False
fast_dev_run = False
