import torch
from torch.utils.data import IterableDataset

from src.ddp.ddp_utils import dist_identity, dprint


def get_shard_info(verbose=False):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        n_shards = dist_identity.world_size
        shard_idx = dist_identity.rank
    else:
        # In a distributed setting, we want to make sure that each worker gets a different set of runs
        n_shards = worker_info.num_workers * dist_identity.world_size
        shard_idx = worker_info.id * dist_identity.world_size + dist_identity.rank
    if verbose:
        dprint(f"get_shard_idx_and_step: Rank {dist_identity.rank} / {dist_identity.world_size-1} | Worker {worker_info.id} / {worker_info.num_workers-1} | Shard {shard_idx}/{n_shards}")
    return shard_idx, n_shards


class DDPIterableDataset(IterableDataset):
    """
    Example of how to implement a custom IterableDataset for use with DDP.
    This class converts a regular Dataset into an IterableDataset.

    This is for reference, to give an example how to implement the __iter__ function
    for both DDP and multiple worker of the DataLoader.

    The class is not useful in itself since the dataset fits in memory to start with.
    Also using Dataloader2 will probably be better,
    check it out: https://pytorch.org/data/beta/dataloader2.html
    """

    def __init__(self, dataset, verbose=False):
        super(DDPIterableDataset, self).__init__()
        if isinstance(dataset, IterableDataset):
            raise ValueError("dataset must not already be an IterableDataset")
        if not hasattr(dataset, "__getitem__"):
            raise ValueError("dataset must have __getitem__")
        if not hasattr(dataset, "__len__"):
            raise ValueError("dataset must have __len__")

        self.dataset = dataset
        self.verbose = verbose

    @property
    def size(self):
        return len(self.dataset)

    def __iter__(self):
        shard_idx, n_shards = get_shard_info(self.verbose)
        for idx in range(self.size):
            if idx % n_shards != shard_idx:
                continue

            # NOTE: If you do some augmentation here, change __len__ accordingly
            #       Copying the data is not necessary, but it is a good idea to do so imo
            #       in case the data is modified in place. I also had cryptic issues during backprop.

            yield self.dataset[idx]

    def __len__(self):
        shard_idx, n_shards = get_shard_info()
        base_len = (self.size - shard_idx + n_shards - 1) // n_shards

        # NOTE: If you do augmentation in the __iter__ function,
        # base_len * augmentation_factor should be returned instead

        return base_len
