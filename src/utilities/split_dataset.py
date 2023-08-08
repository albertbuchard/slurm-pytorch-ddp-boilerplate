from torch.utils.data import Dataset, IterableDataset

from src.ddp.ddp_iterable_dataset import DDPIterableDataset


class SplitDataset:

    def __init__(self, dataset, train, val, test):
        self.dataset = dataset
        self.train = train
        self.val = val
        self.test = test

    def to_ddp_iterable(self):
        for dataset in ["dataset", "train", "val", "test"]:
            d = getattr(self, dataset)
            if isinstance(d, Dataset) and not isinstance(d, IterableDataset):
                setattr(self, dataset, DDPIterableDataset(dataset))

    # def batch_padding(self, val):
    #     self.dataset.batch_padding = val
    #     self.train.batch_padding = val
    #     self.val.batch_padding = val
    #     self.test.batch_padding = val
    #
    # def auto_batch_padding(self, train_batch_size, test_batch_size):
    #     if train_batch_size > 1:
    #         self.train.batch_padding = True
    #     if test_batch_size > 1:
    #         self.test.batch_padding = True
    #         self.val.batch_padding = True
    #     if any([self.train.batch_padding, self.val.batch_padding, self.test.batch_padding]):
    #         self.dataset.batch_padding = True
