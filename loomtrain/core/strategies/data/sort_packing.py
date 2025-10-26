from typing import List
import torch.utils.data as tud

from loomtrain.core.distributed_sampler import (
    DistributedSampler, DistributedBucketSampler
)

from loomtrain.core.data.dataset.base import CollateDataset

from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.core.strategy import DataConfig, DataStrategy
from loomtrain.core.data.dataloader.iter import LoomDataIter


class SortPackingStrategy(DataStrategy):
    def __init__(self,
                 parallel_config: "parallel.ParallelConfig",
                 data_config: "DataConfig",
                 full_determinism: "bool" = False,
                 seed:int = 42
                 ):
        super().__init__(parallel_config = parallel_config, 
                         data_config = data_config,
                         full_determinism = full_determinism, 
                         seed = seed)

    def setup_data_iter(self, dataset: "CollateDataset") -> "LoomDataIter":

        dataset.initialize() # initilize here rather than when datamodule connect strategy, allowing distributed initializing.

        sampler = DistributedSampler(
            dataset,
            bucket_size = self.data_config.packing_length,
            num_replicas = self.num_replicas,
            rank = self.rank,
            shuffle = self.data_config.shuffle,
            seed = self.seed,
            drop_last = self.data_config.drop_last,
            drop_exceed = self.data_config.drop_exceed
        )

        return LoomDataIter(
            dataset, 
            batch_size = self.data_config.batch_size,
            num_epochs = self.data_config.num_epochs,
            collate_fn = dataset.collate_fn,
            pin_memory = self.data_config.pin_memory,
            batch_sampler = sampler
        )