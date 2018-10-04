import json

ATARI = ['Pong-v0', 'Breakout-v0']


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
        self.tau = file["tau"]
        self.max_norm = file["max_norm"]
        self.save_dir = file["save_dir"]
        self.use_tpu = file["use_tpu"]
        self.file_prefix = file["file_prefix"]
