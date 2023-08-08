import os

from wandb.wandb_torch import torch

from src.ddp.ddp_utils import dist_identity


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def size_of_model(model):
    if dist_identity.rank > 0:
        return None

    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp" + str(os.getpid()) + ".pt")
    torch.save(model.state_dict(), full_path)
    size = os.path.getsize(full_path) / (1024 ** 2)
    os.remove(full_path)
    return size