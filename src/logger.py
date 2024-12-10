import wandb


class WandbLogger:
    def __init__(self, config, project_name, logs_path, notes=None):
        self.project_name = project_name
        self.config = config
        self.logs_path = logs_path
        self.notes = notes

    def init(self):
        wandb.init(
            project=self.project_name,
            config=self.config,
            dir=self.logs_path,
            notes=self.notes
        )
