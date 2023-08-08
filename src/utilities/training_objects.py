from dataclasses import dataclass

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


@dataclass
class TrainingObjects:
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    model: nn.Module
    optimizer: Optimizer
    scheduler: LRScheduler