import json


class Config:

    def __init__(self, path):
        with open(path) as f:
            file = json.load(f)

        self.env = file["environment"]
        self.policy = file["policy"]
        self.processes = file["processes"]
        self.steps = file["steps"]
        self.lr = file["lr"]
        self.gamma = file["gamma"]
        self.max_norm = file["max_norm"]
        self.batch_size = file["batch_size"]
        self.device = file["device"]
        self.save_dir = file["save_dir"]
        self.file_prefix = file["file_prefix"]
