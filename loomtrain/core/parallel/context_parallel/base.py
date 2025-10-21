import torch.distributed as dist
from loomtrain.core.strategy import TrainStrategy
from loomtrain.core.parallel import parallel_state as parallel

class ContextParallel:
    def prepare_input(self, *args, **kwargs):
        raise NotImplementedError