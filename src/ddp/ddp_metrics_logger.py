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

    def log_averages_to_wandb(self, keys=None, step=None, reset=True, capitalize=True, prefix=None,
                              exclude=None, include=None):
        self.synchronize(keys=keys)
        if keys is None:
            keys = self.logs.keys()
        wandb = DistributedWandb()
        for key in keys:
            if key not in self.logs:
                continue
            if exclude is not None and any([ex in key for ex in exclude]):
                continue
            if include is not None and not any([inc in key for inc in include]):
                continue
            if len(self.logs[key]) == 0:
                continue
            average = sum(self.logs[key]) / len(self.logs[key])
            # Key to Title
            title = key
            if capitalize:
                title = " ".join([word.capitalize() for word in key.split("_")])
            title = f"{prefix or ''}{title}"
            wandb.log({title: average}, step=step)
            if reset:
                self.logs[key] = []

    def __getitem__(self, key):
        return self.logs[key]

    def synchronize(self, keys=None):
        if not dist.is_initialized() or dist_identity.world_size == 1:
            return

        for source in range(0, dist_identity.world_size):
            safe_barrier()
            objects = [self.logs]
            dist.broadcast_object_list(objects, src=source)
            source_logs = objects[0]
            if source == dist_identity.rank:
                continue
            for key, value in source_logs.items():
                if keys is not None and key not in keys:
                    continue
                self.logs[key].extend(value)

    def get_lasts(self, keys=None, exclude=None, include=None):
        if keys is None:
            keys = self.logs.keys()
        lasts = {}
        for key in keys:
            if len(self.logs[key]) == 0:
                continue
            if exclude is not None and any([ex in key for ex in exclude]):
                continue
            if include is not None and not any([inc in key for inc in include]):
                continue
            lasts[key] = self.logs[key][-1]
        return lasts



metrics_logger = DDPMetricsLogger()
