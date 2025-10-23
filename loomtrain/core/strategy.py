import os, torch, transformers
from typing import Callable
import torch.utils.data as tud
import torch.distributed as dist
from datetime import timedelta
from loomtrain.dataset.base import CollateDataset
from loomtrain.core.data.dataloader.iter import LoomDataIter

# from loomtrain.core.device.mesh import DeviceMes
from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.core.module import LoomModule
from loomtrain.core.actor import LoomActorGroup
from dataclasses import dataclass
from functools import partial


@dataclass
class DataConfig:
    collate_type: Literal["packing", "padding"]
    packing_length: int = None # work while collate_type is packing
    val_interval: int 

class DataStrategy: 
    '''
    prepare dataloader (This class is mainly designed for different data packing algorithms)
    '''
    def __init__(self,
                 parallel_config:"parallel.ParallelConfig",
                 data_config: "DataConfig",
                 full_determinism: "bool" = False,
                 seed:int = 42):
        assert parallel.is_initialized()
        self.parallel_config = parallel_config
        self.data_config = data_config
        self.full_determinism = full_determinism
        self.seed = seed

        self.dp_size = self.parallel_config.dp
        self.num_replicas = self.parallel_config.size_expect_dp

        self.rank = parallel.get_dp_rank()


    def setup_data_iter(self, 
                        dataset: "tud.Dataset") -> "LoomDataIter":
        
        '''
        batch_size is actually micro_batch_size
        '''
        raise NotImplementedError

    @property
    def total_train_steps(self):
        raise NotImplementedError

# TBD
class TrainStrategy:
    '''
    Prepare model, optimizer, scheduler
    '''

    def __init__(self,
                 parallel_config: "parallel.ParallelConfig",
                 init_timeout = timedelta(minutes = 60),
                 full_determinism: bool = False,
                 seed: int = 42,):
        self.parallel_config = parallel_config
        self.init_timeout = init_timeout
        self.full_determinism = full_determinism
        self.seed = seed
            
        self.batch_size = parallel_config.train_batch_size
        self.micro_batch_size = parallel_config.micro_batch_size
        self.grad_accum = parallel_config.grad_accum

    def connect_opt_groups(self, opt_groups: "dict[str, LoomActorGroup]"):
        self.opt_groups = opt_groups
        

    def setup_distributed(self):
        self.set_seed()
        self.set_device()
        self.init_distributed()
        self.init_parallel()
    

    def init_distributed(self):
        raise NotImplementedError
    
    def loomModule_save_ckpt(self, save_dir: str, tag: str):
        raise NotImplementedError

    def loomModule_load_ckpt(self, saved_dir: str, tag: str):
        raise NotImplementedError

    def loomModule_save_module(self, save_dir: str, tag: str):
        raise NotImplementedError

    def loomModule_setup_module(self, modules: "list[nn.Module]" | "dict[str, nn.Module]") -> None:
        raise NotImplementedError

    def loomModule_setup_optimizer(self):
        raise NotImplementedError

    def loomModule_setup_scheduler(self):
        raise NotImplementedError

    def loomModule_backward(self):
        raise NotImplementedError

    def loomModule_step(self):
        raise NotImplementedError


    def loomModule_zero_grad(self):
        raise NotImplementedError

    def config_loomModule_method(self, module: "LoomModule"):
        module.save_ckpt = self.loomModule_save_ckpt
        module.load_ckpt = self.loomModule_load_ckpt
        module.get_saved_sub_dir = lambda : "ckpts"
        module.save_module = self.loomModule_save_module

        try:
            module.setup_module(None)
        except NotImplementedError:
            module.setup_module = self.loomModule_setup_module
        except Exception as e: ...
        try:
            module.backward(None, None)
        except NotImplementedError:
            module.backward = self.loomModule_backward
        except Exception as e: ...
        try:
            module.step()
        except NotImplementedError:
            module.step = self.loomModule_step
        except Exception as e: ...

        
        try:
            module.zero_grad()
        except NotImplementedError:
            module.zero_grad = self.loomModule_zero_grad
        except Exception as e: ...


    def init_parallel(self):
        parallel.initialize(self.parallel_config)

    @property
    def world_size(self):
        if not hasattr(self, "_world_size"):
            self._world_size = dist.get_world_size()
        return self._world_size

    @property
    def rank(self):
        if not hasattr(self, "_rank"):
            self._rank = dist.get_rank()
        return self._rank

    def get_local_rank(self):
        return int(os.environ["LOCAL_RANK"])
    
    def get_local_world_size(self):
        return int(os.environ["LOCAL_WORLD_SIZE"])

    def set_seed(self):
        if self.full_determinism:
            transformers.enable_full_determinism(self.seed)
            transformers.modeling_flash_attention_utils.deterministic_g = True
        else:
            transformers.set_seed(self.seed)
    
    def set_device(self):
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))



    def set_submodule(self, 
                      module: torch.nn.Module, 
                      submodule_name: str,
                      submodule: torch.nn.Module):
        submodules = module.__dict__.get("_modules")
        submodules[submodule_name] = submodule
        module.__dict__[submodule_name] = submodule


    def prepare_train(self, model, optimizer, scheduler):
        return model, optimizer, scheduler
    
    def prepare_eval(self, model):
        return model


    # def backward(self, loss: torch.Tensor, model: "LoomModule"):
    #     model.model.backward(loss)
    
    # def optimizer_step(self, model: "LoomModule"):
    #     model.model.step()


    # def zero_grad(self, model: "LoomModule"):
    #     engine = model.model
        
    #     if engine.bfloat16_enabled():
    #         # TODO: Temporary until bf16_optimizer and zero_optimizer are integrated
    #         if engine.zero_optimization() and hasattr(engine.optimizer, "zero_grad"):
    #             engine.optimizer.zero_grad()
    #         else:
    #             pass
    #     elif engine.zero_optimization() or engine.fp16_enabled() or engine.amp_enabled():
    #         engine.optimizer.zero_grad()
    #     else:
    #         engine.zero_grad()
