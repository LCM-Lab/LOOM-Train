from typing import Iterable
import torch.utils.data as tud
from loomtrain.core.data.dataset.base import CollateDataset


class LoomDataIter(tud.DataLoader):
    def __init__(self, 
                 dataset: "CollateDataset",
                 batch_size: "int" = 1,
                 num_epochs: "int" = 1,
                 shuffle: "bool" | "None" = None, 
                 sampler: "Iterable" | "None" = None,
                 batch_sampler: "Iterable" | "None" = None,
                 num_workers: "int" = 0,
                 pin_memory: "bool" = False,
                 drop_last: "bool" = False,
                 ):
        super().__init__(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            sampler = sampler,
            batch_sampler = batch_sampler,
            num_workers = num_workers,
            collate_fn = dataset.collate_fn,
            pin_memory = pin_memory,
            drop_last = drop_last
        )

        self.num_epochs = num_epochs
        self._exhausted = False

        self.data_iter = iter(self)
        self.next_batch = next(self.data_iter)

    @property
    def exhausted(self):
        return self._exhausted

    def __next__(self):
        current_batch = self.next_batch
        try: self.next_batch = next(self.data_iter)
        except StopIteration: self._exhausted = True
        return current_batch


    def __iter__(self):
        for _ in range(self.num_epochs):
            yield from iter(super().__iter__())