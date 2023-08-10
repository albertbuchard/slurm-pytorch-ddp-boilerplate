from collections import defaultdict

import torch.distributed as dist

from src.ddp.ddp_utils import dist_identity, safe_barrier
from src.ddp.distributed_wandb import DistributedWandb


class DDPMetricsLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(DDPMetricsLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, **kwargs):
        self.logs = defaultdict(list)
        for key, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, list):
                self.logs[key] = value
            else:
                self.logs[key].append(value)

    def store(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                kwargs.update(arg)
            else:
                raise ValueError(f"Expected dict, or named arguments, got positional arg: {arg}")
        for key, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, dict):
                to_store = {
                    f"{key}_{k}": v for k, v in value.items()
                }
                self.store(**to_store)
            elif isinstance(value, list):
                self.logs[key].extend(value)
            else:
                self.logs[key].append(value)

    def log_averages_to_wandb(self, keys=None, step=None, reset=True, capitalize=True):
        if keys is None:
            keys = self.logs.keys()
        wandb = DistributedWandb()
        for key in keys:
            average = sum(self.logs[key]) / len(self.logs[key])
            # Key to Title
            title = key
            if capitalize:
                title = " ".join([word.capitalize() for word in key.split("_")])
            wandb.log({title: average}, step=step)
            if reset:
                self.logs[key] = []

    def __getitem__(self, key):
        return self.logs[key]

    def synchronize(self, keys=None):
        for source in range(0, dist_identity.world_size):
            safe_barrier()
            objects = [self.logs]
            logs_json = dist.broadcast_object_list(objects, source=source)
            source_logs = logs_json[0]
            if source == dist_identity.rank:
                continue
            for key, value in source_logs.items():
                if keys is not None and key not in keys:
                    continue
                self.logs[key].extend(value)


metrics_logger = DDPMetricsLogger()
