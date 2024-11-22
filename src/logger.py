import wandb


class WandbLogger:
    def __init__(self, config, project_name, logs_path):
        self.project_name = project_name
        self.config = config
        self.logs_path = logs_path

    def init(self):
        wandb.init(
            project=self.project_name,
            config=self.config,
            dir=self.logs_path
        )

    def log(self, metric_name, metric):
        wandb.log({metric_name: metric})
