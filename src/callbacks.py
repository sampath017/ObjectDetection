class OverfitCallback:
    def __init__(self, limit_batches=2, limit_train_batches=2, limit_val_batches=2, max_epochs=200):
        if limit_batches > 0:
            self.limit_train_batches = limit_batches
            self.limit_val_batches = limit_batches
        else:
            self.limit_train_batches = limit_train_batches
            self.limit_val_batches = limit_val_batches

        self.max_epochs = max_epochs
