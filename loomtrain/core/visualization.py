import os, wandb
from dataclasses import dataclass
import torch
from torch.utils.tensorboard import SummaryWriter

from loomtrain.core.state import CheckpointMixin
from loomtrain.core.utils import (
    IO, rank0only_decorator, rank0print,
)


@dataclass
class WandbConfig:
    api_key: str
    entity : str
    project: str
    group  : str
    name   : str
    config : dict
    reinit : bool = True


@dataclass
class TensorboardConfig:
    log_dir: str
    name : str



class VisualizationModule(CheckpointMixin):
    def __init__(self,
                 tensorboard_config: "TensorboardConfig" = None,
                 wandb_config: "WandbConfig" = None):
        
        assert tensorboard_config is not None or wandb_config is not None, \
            "You should assign at least one visualization method !"

        self._init_tensorboard(tensorboard_config)
        self._init_wandb(wandb_config)

    @rank0only_decorator
    def _init_tensorboard(self, tensorboard_config: TensorboardConfig):
        self.tensorboard_config = tensorboard_config
        if tensorboard_config:
            IO.mkdir(tensorboard_config.log_dir)
            log_dir = os.path.join(tensorboard_config.log_dir, tensorboard_config.name)
            self._tensorboard = SummaryWriter(log_dir = log_dir)
    
    @rank0only_decorator
    def _update_tensorboard(self, logs_dict:dict, global_step: int, logging_steps:int = 1):
        if self.tensorboard_config and global_step % logging_steps == 0:
            for k, v in logs_dict.items():
                self._tensorboard.add_scalar(k, v, global_step)
            

    @rank0only_decorator
    def _release_tensorboard(self):
        if self.tensorboard_config:
            self._tensorboard.close()

    @rank0only_decorator
    def _init_wandb(self, wandb_config: WandbConfig):
        self.wandb_config = wandb_config
        if wandb_config:
            wandb.login(key = wandb_config.api_key)
            wandb.init(
                entity = wandb_config.entity,
                project = wandb_config.project,
                group = wandb_config.group,
                name = wandb_config.name,
                config = wandb_config.config,
                reinit = wandb_config.reinit
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", 
                                step_metric = "train/global_step",
                                step_sync = True)
            
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", 
                                step_metric = "eval/global_step",
                                step_sync = True)
    
    @rank0only_decorator
    def _update_wandb(self, logs_dict:dict, global_step:int, logging_steps:int = 1):
        if self.wandb_config and global_step % logging_steps == 0:
            wandb.log(logs_dict, step = global_step)


    @rank0only_decorator
    def _release_wandb(self):
        if self.wandb_config:
            wandb.finish()


    @rank0only_decorator
    def update(self, logs_dict:dict):
        global_step = logs_dict.pop('global_step')
        logging_steps = logs_dict.pop('logging_steps')
        self._update_tensorboard(logs_dict, global_step, logging_steps)
        self._update_wandb(logs_dict, global_step, logging_steps)


    def get_saved_sub_dir(self): return "tensorabord"

    @rank0only_decorator
    def save_ckpt(self, save_dir, tag):
        #TODO: save in save_dir/tensorboard
        return super().save_ckpt(save_dir, tag)

    @rank0only_decorator
    def load_ckpt(self, saved_dir, tag):
        #TODO: load from saved_dir/tensorborad
        return super().load_ckpt(saved_dir, tag)

    @rank0only_decorator
    def release(self):
        self._release_tensorboard()
        self._release_wandb()



class Accumulator:
    def __init__(self, value = 0, total = 0):
        self.value = value
        self.total = total
    def __iadd__(self,other: "Accumulator"):
        self.value += other.value
        self.total += other.total
        return self

    def __add__(self, other: "Accumulator"):
        return Accumulator(self.value + other.value,
                           self.total + other.total)

    def reset(self):
        self.value = 0
        self.total = 0

    def get_value(self):
        if self.total == 0: return None
        value = self.value
        if isinstance(value, torch.Tensor):
            value = value.item()
        return value/self.total
    
