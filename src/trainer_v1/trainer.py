import json
import os

import torch
from torch.nn import functional as F
from tqdm import tqdm

from src.ddp.ddp_utils import dprint, dist_identity, distribute_str, conditional_join
from src.ddp.distributed_wandb import DistributedWandb
from src.ddp.device_singleton import device
from src.trainer_v1.factories import get_training_objects
from src.utilities import current_config


def train(training_items, epoch, dry_run=False):
    model = training_items.model
    train_loader = training_items.train_loader
    optimizer = training_items.optimizer
    _device = device.get()

    model.train()
    batch_idx = 0
    losses = []
    with conditional_join(model, optimizer):
        pbar = tqdm(train_loader, desc=f"Rank 0 | Training", position=2,
                    leave=False, colour="blue", disable=dist_identity.rank != 0)
        for (data, target) in pbar:
            data, target = data.to(_device), target.to(_device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            batch_idx += 1
            losses.append(loss.item())
            # update tqdm postfix
            pbar.set_postfix({"Epoch": epoch, "Loss": loss.item()})
            if dry_run:
                break
        pbar.close()

    training_loss = None
    if len(losses) > 0:
        training_loss = sum(losses) / len(losses)

    if current_config.stdout:
        dprint(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {training_loss:.6f}', flush=True)

    return training_loss


def validate(training_items, type="validation"):
    model = training_items.model
    data_loader = training_items.validation_loader if type == "validation" else training_items.test_loader
    if data_loader is None:
        return None
    losses = []
    _device = device.get()
    with conditional_join(model), torch.no_grad():
            pbar = tqdm(data_loader, desc=f"Rank 0 | {type.capitalize()}",
                        position=2, leave=False,
                        colour="yellow", disable=dist_identity.rank != 0)
            for (data, target) in pbar:
                data, target = data.to(_device), target.to(_device)
                output = model(data)
                losses.append(F.nll_loss(output, target, reduction='sum').item())
            pbar.close()

    average_loss = None
    if len(losses) > 0:
        average_loss = sum(losses) / len(losses)

    if current_config.stdout:
        dprint(f'\n{type.capitalize()} set: Average loss: {average_loss:.4f}\n', flush=True)

    return average_loss


def training_loop(dry_run=False, no_save=False):
    # Get training items
    training_items = get_training_objects()

    # Wandb watch model
    wandb = DistributedWandb()
    if not device.is_mps():
        # Wandb watch model does not work seem to work well with MPS
        wandb.watch(training_items.model)

    # Training loop, conditional join is used to make sure that all processes
    # are in sync when one process is done before the others (avoids process hanging forever)
    with conditional_join(training_items.model, training_items.optimizer):
        for epoch in tqdm(range(current_config["max_epochs"]), desc="Epochs", position=1, leave=False, colour="green",
                          disable=dist_identity.rank != 0):
            training_loss = train(training_items=training_items, epoch=epoch, dry_run=dry_run)
            validation_loss = validate(training_items, type="validation")
            training_items.scheduler.step()
            wandb.log({"Training Loss": training_loss,
                       "Validation Loss": validation_loss,
                       "Epoch": epoch,
                       "lr": training_items.scheduler.get_last_lr()[0]})

        if training_items.test_loader is not None:
            test_loss = validate(training_items, type="test")
            wandb.log({"Test Loss": test_loss})

    if not no_save and dist_identity.rank == 0:
        run_dir = wandb.run_dir if wandb.run_dir is not None else os.path.join(current_config.project_root, "models")
        os.makedirs(run_dir, exist_ok=True)
        model_name = f'{current_config.get("project", "default")}_{current_config.hash}.pt'
        torch.save(training_items.model.state_dict(), os.path.join(run_dir, model_name))
        dprint("Saved model")


def sweep_training_loop():
    # Initialize wandb for Rank 0 (necessary to get the run config)
    wandb = DistributedWandb().init()
    # Convert run config to json
    # and distribute it from Rank 0 to all other ranks
    config_json = None
    if wandb.config is not None:
        config_json = json.dumps({k: v for k, v in wandb.config.items()})
    new_config = distribute_str(config_json, max_length=10000)
    # Update config with sweep config on all ranks
    current_config.update(json.loads(new_config))
    # Run training loop on all ranks
    training_loop(no_save=True)


def sweep():
    wandb = DistributedWandb()

    # Initialize wandb sweep and shares sweep_id with all ranks
    # that is sweep_id will be set in all processes to the same value here,
    # it is not really needed but just in case
    sweep_id = wandb.sweep(current_config.sweep_config)

    # This only runs the sweep on Rank 0, other ranks will run the function directly
    wandb.agent(sweep_id, function=sweep_training_loop)
    # dprint("Sweep finished")
