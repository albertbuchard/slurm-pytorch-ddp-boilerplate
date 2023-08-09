import argparse
import os
import sys
import time

import torch
from dotenv import load_dotenv

root = os.path.realpath(__file__).split("slurm-pytorch-ddp-boilerplate")[0]
project_root = os.path.join(root, "slurm-pytorch-ddp-boilerplate")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.trainer_v1.configurations.default_configuration import default_model_config, default_data_config, \
    default_trainer_config, default_wandb_config, default_sweep_config
from src.trainer_v1.trainer import sweep, training_loop
from src.utilities import current_config

from src.ddp.ddp_utils import device, dprint, ddp_setup, dist_identity, safe_barrier, hello_distributed
from src.ddp.distributed_wandb import DistributedWandb

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description='BCause TorchRun')
    # Training
    parser.add_argument('--batch-size', type=int, default=current_config["batch_size"], metavar='N',
                        help=f'input batch size for training (default: {current_config["batch_size"]})')
    parser.add_argument('--test-batch-size', type=int, default=current_config["test_batch_size"], metavar='N',
                        help=f'input batch size for testing (default: {current_config["test_batch_size"]})')
    parser.add_argument('--epochs', type=int, default=current_config["max_epochs"], metavar='N',
                        help=f'number of epochs to train (default: {current_config["max_epochs"]})')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # DDP
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='disable CUDA/MPS (default: False)')
    parser.add_argument('--backend', type=str, default=None,
                        help='Distributed backend (default: gloo if --cpu, nccl otherwise)')
    # Wandb
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb', default=False)
    parser.add_argument('--wandb-all-processes', action='store_true', help='Wandb emits on all processes',
                        default=False)
    parser.add_argument('--wandb-project', type=str, help='Wandb project name',
                        default=current_config.wandb_config["project"])
    parser.add_argument('--wandb-group', type=str, help='Wandb group', default=None)
    parser.add_argument('--sweep-config', type=str, help='Wandb sweep config', default=None)
    # Is wandb sweep
    parser.add_argument('--sweep', action='store_true', help='Run sweep', default=False)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Should run on CPU?
    cpu = args.cpu or (not torch.cuda.is_available() and not torch.has_mps)

    # Setup distributed backend
    backend = args.backend
    if backend is not None and cpu:
        dprint("Backend specified, but CPU specified too. Backend set to gloo.")
        backend = "gloo"
    elif backend not in ["gloo", "nccl"]:
        dprint("Backend not specified, or invalid.  Ignoring.")
        backend = None
    if backend is None:
        backend = "gloo" if cpu else "nccl"

    if cpu:
        dprint("Disabling CUDA/MPS")
        device.set("cpu")

    # Store config
    current_config["cpu_only"] = cpu
    current_config["max_epochs"] = args.epochs
    current_config["batch_size"] = args.batch_size
    current_config["test_batch_size"] = args.test_batch_size
    current_config["project"] = args.wandb_project

    # Setup DDP
    if dist_identity.ddp_available:
        ddp_setup(backend)
    elif not cpu and torch.has_mps:
        device.set("mps")

    hello_distributed()

    # Setup wandb
    if args.sweep:
        if args.sweep_config is not None:
            current_config.load(args.sweep_config, "sweep_config")
        # Wandb necessary for sweep
        args.no_wandb = False
        # During sweep, wandb emits only on rank 0
        args.wandb_all_processes = False

    # Initialize distributed wandb
    wandb = DistributedWandb(project=args.wandb_project, group=args.wandb_group, every_process=args.wandb_all_processes)

    if args.no_wandb:
        # This disables the DistributedWandb singleton.
        # When wandb.is_emitting is set to False, all calls to wandb will have no effect
        # (this allows us to keep wandb calls in the code without them doing anything).
        dprint("Disabling wandb")
        wandb.is_emitting = False

    # NOTE: If you do some heavy compute on Rank 0 here
    # Might be a good idea to wait for all processes to be ready,
    # may not be necessary, but you can use:
    #   safe_barrier()

    # Training
    if args.sweep:
        sweep()
    else:
        wandb.init()
        training_loop(args.dry_run)

    wandb.finish()


if __name__ == '__main__':
    dprint("Main called")
    dprint("sys.path:", sys.path)
    dprint("torch.__file__:", torch.__file__)
    dprint("torch.__version__:", torch.__version__)

    # Setup default config in the current_config singleton
    current_config.reset(
        root=root,
        project_root=project_root,
        model_config=default_model_config,
        data_config=default_data_config,
        trainer_config=default_trainer_config,
        wandb_config=default_wandb_config,
        sweep_config=default_sweep_config,
        stdout=dist_identity.rank == 0
    )

    start = time.time()
    try:
        main()
    except Exception as e:
        dprint(f"Exception: {e}")
        raise e
    finally:
        end = time.time()
        dprint(f"Time taken: {end - start} seconds")
        safe_barrier()
        if dist_identity.rank == 0:
            dprint("Destroying process group")
            # dist.destroy_process_group() TODO not necessary? creates error
