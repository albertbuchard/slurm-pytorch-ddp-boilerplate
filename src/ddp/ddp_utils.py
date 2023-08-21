import json
import os
from contextlib import contextmanager
from socket import gethostname

import torch
from torch import distributed as dist
from torch.distributed import init_process_group
from torch.distributed.algorithms.join import Join, Joinable
from torch.nn.parallel import DistributedDataParallel

from src.ddp.device_singleton import device


class DistributedIdentity:
    """
    Singleton class to hold distributed identity information.
    Handles SLURM, torchrun, and local runs.

    Looks for the following environment variables:
    - RANK
    - WORLD_SIZE
    - LOCAL_RANK
    - SLURM_PROCID
    - SLURM_NTASKS
    - SLURM_LOCALID
    - SLURM_GPUS_PER_NODE
    - SLURMD_NODENAME
    - SLURM_CPUS_PER_TASK
    - GROUP_RANK
    - LOCAL_WORLD_SIZE
    - TORCHELASTIC_RESTART_COUNT
    - TORCHELASTIC_MAX_RESTARTS
    - TORCHELASTIC_RUN_ID
    - MASTER_ADDR
    - MASTER_PORT

    If SLURM_JOB_ID is set, then the job_id will be set to that value.
    Otherwise, the job_id will be set to "local-" + str(os.getpid())

    If RANK is set, then rank will be set to that value.
    Otherwise, if SLURM_PROCID is set, then rank will be set to that value.
    Otherwise, rank will be set to 0.

    If WORLD_SIZE is set, then world_size will be set to that value.
    Otherwise, if SLURM_NTASKS is set, then world_size will be set to that value.
    Otherwise, world_size will be set to 1.

    If LOCAL_RANK is set, then local_rank will be set to that value.
    Otherwise, if SLURM_LOCALID is set, then local_rank will be set to that value.

    If SLURMD_NODENAME is set, then nodename will be set to that value.
    Otherwise, nodename will be set to the result of socket.gethostname()
    """
    _instance = None
    rank = None
    gpus_per_node = None
    local_rank = None
    world_size = None
    nodename = None
    cpu_per_task = None

    local_world_size = None
    master_addr = None
    master_port = None
    is_torchelastic = None
    torch_restart_count = None
    torch_max_restarts = None
    torch_runid = None
    group_rank = None
    job_id = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(DistributedIdentity, cls).__new__(cls, *args, **kwargs)
            if os.environ.get("SLURM_JOB_ID", None) is not None:
                cls.job_id = str(os.environ["SLURM_JOB_ID"])
            else:
                cls.job_id = "local-" + str(os.getpid())
            if os.environ.get("RANK", None) is not None:
                cls.rank = int(os.environ["RANK"])
            elif os.environ.get("SLURM_PROCID", None) is not None:
                cls.rank = int(os.environ["SLURM_PROCID"])
            else:
                cls.rank = 0
            if os.environ.get("WORLD_SIZE", None) is not None:
                cls.world_size = int(os.environ["WORLD_SIZE"])
            elif os.environ.get("SLURM_NTASKS", None) is not None:
                cls.world_size = int(os.environ["SLURM_NTASKS"])
            else:
                cls.world_size = 1
            if os.environ.get("LOCAL_RANK", None) is not None:
                cls.local_rank = int(os.environ["LOCAL_RANK"])
            elif os.environ.get("SLURM_LOCALID", None) is not None:
                cls.local_rank = int(os.environ["SLURM_LOCALID"])
            elif cls.rank is not None and cls.world_size is not None:
                cls.local_rank = cls.rank % cls.world_size
            if os.environ.get("SLURM_GPUS_PER_NODE", None) is not None:
                cls.gpus_per_node = int(os.environ["SLURM_GPUS_PER_NODE"])
            if os.environ.get('SLURMD_NODENAME', None) is not None:
                cls.nodename = os.environ["SLURMD_NODENAME"]
            else:
                cls.nodename = gethostname()

            if os.environ.get("GROUP_RANK", None) is not None:
                cls.group_rank = int(os.environ["GROUP_RANK"])
            if os.environ.get("LOCAL_WORLD_SIZE", None) is not None:
                cls.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            if os.environ.get("TORCHELASTIC_RESTART_COUNT", None) is not None:
                cls.torch_restart_count = int(os.environ["TORCHELASTIC_RESTART_COUNT"])
            if os.environ.get("TORCHELASTIC_MAX_RESTARTS", None) is not None:
                cls.torch_max_restarts = int(os.environ["TORCHELASTIC_MAX_RESTARTS"])

            cls.is_torchelastic = False
            if (os.environ.get("TORCHELASTIC_RUN_ID", None) is not None and
                    os.environ.get("TORCHELASTIC_RUN_ID", None) != "" and
                    os.environ.get("TORCHELASTIC_RUN_ID", None) != "none"):
                cls.is_torchelastic = True
                print("TORCHELASTIC_RUN_ID found. Setting torch_runid to that value.",
                      os.environ["TORCHELASTIC_RUN_ID"])
                cls.torch_runid = int(os.environ["TORCHELASTIC_RUN_ID"])

            if os.environ.get("SLURM_CPUS_PER_TASK", None) is not None:
                cls.cpu_per_task = int(os.environ["SLURM_CPUS_PER_TASK"])
            elif cls.is_torchelastic:
                print("SLURM_CPUS_PER_TASK not found. Using max(1, (os.cpu_count() // cls.world_size) - 1)")
                print(f"os.cpu_count(): {os.cpu_count()}")
                print(f"cls.world_size: {cls.world_size}")
                print(
                    f"max(1, (os.cpu_count() // cls.world_size) - 1): {max(1, (os.cpu_count() // cls.world_size) - 1)}")
                cls.cpu_per_task = max(1, (os.cpu_count() // cls.world_size) - 1)

            cls.master_addr = os.environ.get('MASTER_ADDR', None)
            cls.master_port = os.environ.get('MASTER_PORT', None)

        return cls._instance

    @property
    def ddp_available(self):
        return (dist.is_available() and self.world_size > 1 and
                (self.is_torchelastic or (self.master_addr is not None and self.master_port is not None)))

    @property
    def is_slurm(self):
        return os.environ.get("SLURM_JOB_ID", None) is not None

    @property
    def available_cpu_count(self):
        if self.cpu_per_task is not None:
            return self.cpu_per_task
        import multiprocessing
        return min(os.cpu_count(), multiprocessing.cpu_count())

    def __str__(self):
        return f"Rank {self.rank} (local rank {self.local_rank}) of {self.world_size} on {self.nodename}"

    def __repr__(self):
        return self.__str__()


dist_identity = DistributedIdentity()


def dprint(*args, **kwargs):
    """
    Prints to stdout with the rank and device prepended.
    Passes arguments through to print.

    :param args:
    :param kwargs:
    :return: None
    """
    device_str = ""
    if device.is_cpu():
        device_str = "CPU | "
    elif device.is_cuda():
        device_str = f"GPU {torch.cuda.current_device()} | "
    elif device.is_mps():
        device_str = f"MPS | "

    log_prefix = f"[{dist_identity.nodename}] Rank {dist_identity.rank} | "
    print(log_prefix, device_str, *args, **kwargs)


def ddp_setup(backend="nccl", force_spawn=False):
    """
    Initializes the distributed group with the specified backend.
    :param backend: either "nccl" or "gloo"
    :param force_spawn: if True, will force the spawn start method to be set, might be necessary for NCCL
    :return: None
    """
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available. DDP is only supported on torch.distributed.")
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. NCCL is only supported on CUDA devices.")
        if dist_identity.local_rank is None:
            raise RuntimeError("LOCAL_RANK is not set. DDP is only supported on torch.distributed.")
        # NCCL backend prefers the spawn start method to be set
        if force_spawn:
            torch.multiprocessing.set_start_method('spawn', force=True)
    elif backend != "gloo":
        raise ValueError("backend must be 'nccl' or 'gloo'")

    dprint(f"Initializing the process group with backend: {backend}", flush=True)
    dprint(" - CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET"))
    dprint(" - LOCAL_RANK:", dist_identity.local_rank)
    dprint(" - IS_TORCHELASTIC:", dist_identity.is_torchelastic)

    if not dist.is_initialized():
        if dist_identity.is_torchelastic:
            init_process_group(backend)
        else:
            init_process_group(backend,
                               rank=dist_identity.rank,
                               world_size=dist_identity.world_size)
        dprint("DDP initialized", flush=True)
    else:
        dprint("DDP already initialized", flush=True)

    safe_barrier()
    str_device = f"cuda:{dist_identity.local_rank}" if backend == "nccl" else "cpu"
    dprint("Setting device:", str_device, flush=True)
    device.set(torch.device(str_device))

    dprint(" - DEVICE:", device, flush=True)


def safe_barrier():
    if dist.is_initialized():
        dist.barrier()


def hello_distributed():
    dprint(f"Hello {dist_identity.__str__()}", flush=True)
    dprint(json.dumps(dist_identity.__dict__, indent=4), flush=True)
    dprint(f"Group initialized: {dist.is_initialized()}", flush=True)
    dprint(f"Device: {str(device)}", flush=True)
    try:
        dprint("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET"))
        dprint("torch.cuda.is_available():", torch.cuda.is_available())
        dprint("torch.cuda.device_count():", torch.cuda.device_count())
        dprint("torch.cuda.current_device():", torch.cuda.current_device())
        dprint("torch.cuda.get_device_name():", torch.cuda.get_device_name())
        dprint("torch.cuda.get_device_capability():", torch.cuda.get_device_capability())
        dprint("torch.cuda.get_device_properties():", torch.cuda.get_device_properties(torch.cuda.current_device()))
    except Exception as e:
        dprint(f"Exception: {e}")


def distribute_str(value, max_length=100):
    """
    Shares the value string from rank 0 to all other processes in a distributed setting.
    If dist is not initialized, this function will raise a RuntimeError for non-zero ranks,
    and will pass through the value for rank 0.

    :param value: The string to share
    :param max_length: The maximum length of the string

    :return: The shared string
    """
    if not dist.is_initialized():
        if dist_identity.rank == 0:
            return value
        raise RuntimeError("Distributed group must be initialized before calling distribute_str")

    if max_length is None or not isinstance(max_length, int):
        raise ValueError("max_length must be an integer")
    if max_length < 1:
        raise ValueError("max_length must be greater than 0")

    # Create a tensor to hold the sweep_id
    value_tensor = torch.zeros([max_length], dtype=torch.int8, device=device.get())

    if dist_identity.rank == 0:
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        if len(value) > max_length:
            raise ValueError("value must be less than max_length")

        # Convert string to tensor (assuming ASCII encoding for simplicity)
        for i, char in enumerate(value):
            value_tensor[i] = ord(char)

    # Broadcast the sweep_id_tensor from rank 0 to all other processes
    dist.broadcast(value_tensor, src=0)

    # Convert tensor back to string for non-zero ranks
    if dist_identity.rank != 0:
        chars = [chr(value_tensor[i].cpu().item()) for i in range(max_length) if value_tensor[i].item() != 0]
        value = ''.join(chars)

    return value


@contextmanager
def conditional_join(model, optimizer=None):
    """
    Context manager that joins the model and optimizer if they are joinable, that is
    if they are distributed. This is useful to avoid hanging when one process is done
    before the others.
    If neither model nor optimizer are joinable, this context manager does nothing.
    Allows running on a single GPU or CPU.

    :param model: nn.Module
    :param optimizer: optim.Optimizer
    :return: None
    """
    context = None
    joinables = []
    if isinstance(model, DistributedDataParallel):
        joinables.append(model)
    if optimizer is not None and isinstance(optimizer, Joinable):
        joinables.append(optimizer)
    if len(joinables) > 0:
        context = Join(joinables)
        context.__enter__()
    try:
        yield
    finally:
        if context:
            context.__exit__(None, None, None)
