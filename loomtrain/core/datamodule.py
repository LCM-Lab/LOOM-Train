
from loomtrain.core.state import CheckpointMixin
from loomtrain.core.strategy import DataStrategy
from loomtrain.core.data.dataset.base import CollateDataset
import torch
from loomtrain.core.parallel import parallel_state as parallel


class LoomDataModule(CheckpointMixin):
    def __init__(self,
                 train_dataset: "CollateDataset",
                 val_dataset: "CollateDataset",
                 ):
        assert parallel.is_initialized(), "One must init `LoomTrainer` before init `LoomDataModule`"

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self._is_training = False
        self._global_step = 0
    
    def to_current_device(self, *args):
        if len(args) == 1:
            if isinstance(args[0], list): return [self.to_current_device(k) for k in args[0]]
            if isinstance(args[0], tuple): return tuple(self.to_current_device(k) for k in args[0])
            if isinstance(args[0], torch.Tensor): return args[0].to(torch.cuda.current_device())
            return args[0]
        return tuple(self.to_current_device(k) for k in args)

    @property
    def is_validating_step(self):
        return self.global_step % self.strategy.data_config.val_interval == 0

    def connect_strategy(self, strategy: "DataStrategy"):
        assert isinstance(strategy, DataStrategy)
        self.strategy = strategy
        self.train_dataset.initialize()
        self.val_dataset.initialize()

    @property
    def total_train_steps(self):
        return len(self.train_data_iter)

    @property
    def total_val_steps(self):
        return len(self.val_data_iter)

    @property
    def exhausted(self) -> bool:
        self.train_data_iter.exhausted

    @property
    def training(self):
        return self._is_training

    def train(self):
        self._is_training = True

    def eval(self):
        self._is_training = False

    def _setup_train_data_iter(self):
        self.train_data_iter = self.strategy.setup_data_iter(self.train_dataset)

    def _setup_val_data_iter(self):
        self.val_data_iter = self.strategy.setup_data_iter(self.val_dataset)

    def _update(self):
        batches = super()._update()
        return self.to_current_device(batches)

    def get_saved_sub_dir(self): return "data_iter"

    def update(self):
        return next(self.train_data_iter)
