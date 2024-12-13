import wandb


class WandbLogger:
    def __init__(self, config, project_name, logs_path, notes=None, offline=False):
        self.project_name = project_name
        self.config = config
        self.logs_path = logs_path
        self.notes = notes
        self.mode = "offline" if offline else "online"

    def init(self):
        wandb.init(
            project=self.project_name,
            config=self.config,
            dir=self.logs_path,
            notes=self.notes,
            mode=self.mode
        )
