import os

import torch
import wandb
from torch import distributed as dist, optim
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.ddp.ddp_utils import dprint, dist_identity
from src.ddp.device_singleton import device
from src.ddp.model_utils import count_parameters, size_of_model
from src.trainer_v1.model import MNISTNet
from src.utilities import current_config, TrainingObjects, SplitDataset


def get_model():
    if current_config["model_config"] is None:
        raise ValueError("Model config is not set")

    # Get model
    model = MNISTNet(current_config["model_config"])

    # Move model to device
    model = model.to(device.get())

    if current_config.stdout:
        dprint(f"Moved model to {device}")
        # Get model size and print
        model_params = count_parameters(model)
        dprint(f"Model has {model_params} parameters")
        model_size = size_of_model(model)
        dprint(f"Model size is {model_size} MB")
        wandb.log({"model/params": model_params})
        wandb.log({"model/size": model_size})

    return model


def get_datasets():
    """
    Get the datasets for training, validation and testing
    :return:
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    data_folder = current_config.get("data_folder", os.path.join(current_config["project_root"], "data"))
    os.makedirs(data_folder, exist_ok=True)

    train_dataset = datasets.MNIST(data_folder, train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(data_folder, train=False,
                                  transform=transform)

    # Split training dataset into training and validation
    # test_size not used here
    val_dataset = None
    if current_config.get("validation_size", 0) > 0:
        num_samples = len(train_dataset)
        train_size = int(num_samples * (1 - current_config["validation_size"]))
        val_size = int(num_samples * current_config["validation_size"])

        # Create indices for the split
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, train_size + val_size))

        # Create deterministic subsets
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(train_dataset, val_indices)

    split_datasets = SplitDataset(None, train_dataset, val_dataset, test_dataset)
    if current_config["use_ddp_iterable_dataset"]:
        # Convert to DDPIterableDataset
        # Just for reference in case you need a custom iterable dataset
        split_datasets.to_ddp_iterable()

    return split_datasets


# For reference - could be used for custom collate_fn
# def collate_fn(batch):
#     return {
#         'input': torch.stack([x[0] for x in batch]),
#         'labels': torch.tensor([x[1] for x in batch]).unsqueeze(-1)
#     }


def get_training_objects():
    datasets = get_datasets()

    num_workers = dist_identity.cpu_per_task
    if num_workers is None:
        num_cores = os.cpu_count()
        num_workers = max(1, min(2, num_cores - 2))
        dprint(f"Using {num_workers} workers for data loading")
        # num_workers = 0

    train_loader = torch.utils.data.DataLoader(datasets.train,
                                               # collate_fn=collate_fn,
                                               batch_size=current_config["batch_size"],
                                               num_workers=num_workers)
    validation_loader = None
    if datasets.val is not None:
        validation_loader = torch.utils.data.DataLoader(datasets.val,
                                                        # collate_fn=collate_fn,
                                                        batch_size=current_config["test_batch_size"],
                                                        num_workers=num_workers)
    test_loader = None
    if datasets.test is not None:
        test_loader = torch.utils.data.DataLoader(datasets.test,
                                                  # collate_fn=collate_fn,
                                                  batch_size=current_config["test_batch_size"],
                                                  num_workers=num_workers)

    # Get model
    if current_config.stdout:
        dprint("Model creation", flush=True)
    model, transformer_config = get_model()
    model.return_self_loss = False
    if current_config.stdout:
        dprint("Model created", flush=True)

    # Setup optimization
    if current_config["optimizer"] == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=current_config["lr"])
    else:
        optimizer = optim.Adam(model.parameters(), lr=current_config["lr"])

    if dist.is_initialized():
        dist.barrier()
        kwargs = {"find_unused_parameters": True}
        if not device.is_cpu() and dist_identity.local_rank is not None:
            kwargs["device_ids"] = [dist_identity.local_rank]
        model = DDP(model, **kwargs)
        optimizer = ZeroRedundancyOptimizer(model.parameters(),
                                            optimizer_class=optim.Adadelta,
                                            lr=current_config["lr"])
        if current_config.stdout:
            dprint("Model converted to DDP", flush=True)

    scheduler = StepLR(optimizer, step_size=1, gamma=current_config["lr_decay"])
    if current_config["lr_scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=current_config["max_epochs"])

    return TrainingObjects(train_loader, validation_loader, test_loader, model, optimizer, scheduler)
