from typing import List
import torch.utils.data as tud
from loomtrain.utils.distributed_sampler import (
    DistributedSampler, DistributedBucketSampler
)




from loomtrain.core.data.dataset.base import CollateDataset

from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.core.strategy import DataStrategy

class DataIterator: ...

class SortPackingStrategy(DataStrategy):
    def __init__(self,
                 parallel_config: "parallel.ParallelConfig",
                 bucket_size: int,
                 batch_size: int = 1,
                 pin_memory: bool = False,
                 shuffle: bool = True,
                 drop_last: bool = True,
                 drop_exceed: bool = False,
                 full_determinism: bool= False,
                 seed:int = 42):
        super().__init__(parallel_config = parallel_config, 
                         full_determinism = full_determinism, 
                         seed = seed)
        self.bucket_size = bucket_size
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.drop_exceed = drop_exceed

    def setup_data_iter(self, dataset: "CollateDataset") -> "DataIterator":
        from loomtrain.core.data.distributed_sampler import DistributedBucketSampler

        sampler = DistributedSampler(
            dataset,
            bucket_size = self.bucket_size,
            num_replicas = self.num_replicas,
            rank = self.rank,
            shuffle = self.shuffle,
            seed = self.seed,
            drop_last = self.drop_last,
            drop_exceed = self.drop_exceed
        )

        return tud.DataLoader(
            dataset, 
            collate_fn = dataset.collate_fn,
            pin_memory = self.pin_memory,
            batch_sampler = sampler
        )

    