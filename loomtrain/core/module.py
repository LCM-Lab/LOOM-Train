import os
from torch import nn
import torch.distributed as dist
from transformers import PreTrainedTokenizer
from collections import defaultdict
from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.core.strategy import TrainStrategy
from loomtrain.core.state import CheckpointMixin
from loomtrain.core.actor import LoomOptDict, LoomActorGroup
from loomtrain.core.visualization import Accumulator
from loomtrain.utils.lora import LoRAConfig, get_peft_model
from dataclasses import dataclass


class LoomModule(CheckpointMixin):
    def __init__(self, opt_dicts: "list[LoomOptDict]" | "dict[str, LoomOptDict]" = None, actor_groups: "list[LoomActorGroup]" | "dict[str, LoomActorGroup]" = None):        
        assert parallel.is_initialized(), "One must init `LoomTrainer` before init `LoomModule`"

        if isinstance(opt_dicts, list):
            opt_dicts = {f"model_{str(i)}": dic for i, dic in enumerate(opt_dicts)}

        self.opt_dicts = opt_dicts

        if isinstance(actor_groups, list):
            actor_groups = {f"model_{str(i)}": dic for i, dic in enumerate(actor_groups)}

        self._actor_groups = actor_groups

    def _validate(self, datadmoule: "LoomDataModule"):
        logs_dict = dict()
        if datadmoule.is_validating_step():
            self.eval()
            datamodule.eval()
            datadmoule._setup_val_data_iter()
            with torch.no_grad():
                logs_dict = self.validate(datadmoule.val_data_iter)
            self.train()
            datamoduel.train()
        return logs_dict

    @property
    def training(self):
        return next(self.opt_dicts.values())["actor"].training
    def train(self):
        for opt_group in self.opt_groups.values():
            opt_group["actor"].train()
    def eval(self):
        for opt_group in self.opt_groups.values():
            opt_group["actor"].eval()

    def connect_datamodule(self, datamodule: "LoomDataModule"):
        '''must be called before connect_strategy, because total_steps unset'''
        self.datamodule = datamodule
        for group in self.opt_dicts:
            group.total_steps = datamodule.total_train_steps

    def connect_strategy(self, strategy: "TrainStrategy"):
        assert parallel.is_initialized()
        assert isinstance(strategy, TrainStrategy)
        self.strategy = strategy
        self.strategy.config_loomModule_method(self)
        if self._actor_groups is None:        
            opt_groups = self.setup_module(self.opt_dicts)
            self.opt_groups = self._setup_actors(opt_groups)
        else: 
            self.opt_groups = self._actor_groups

        self.strategy.connect_opt_groups(self.opt_groups)
        self.setup_self_module()
        self.zero_grad()


    def _save_module(self, checkpoint_config: "CheckpointConfig"):
        if checkpoint_config.weight_interval % self.global_step: return

        save_dir = os.path.join(checkpoint_config.save_dir, "models", f"global_step{self.global_step}")
        if dist.get_rank() == 0:
            os.makedirs(save_dir, exist_ok = True)
        dist.barrier()

        self.save_module(save_dir)
        
        dist.barrier()
        torch.cuda.synchronize()

        if dist.get_rank() == 0:
            print(f"Model Weight: {save_dir} is ready !!!")
    
    def _setup_actors(self, opt_groups: "dict[str, LoomActorGroup]") -> "dict[str, LoomActorGroup]":
        for group in opt_groups.values():
            group.build_actor()

    def setup_self_module(self):
        '''this function will be called after all LoomActorGroups have be setup.'''
        ...

    def save_module(self, save_dir: str):
        '''
        save_dir is already be set different
        This Function can either be implemented manually, or be replaced by train_strategy'''
        raise NotImplementedError


    def setup_module(self, opt_dicts: "dict[str, LoomOptDict]") -> "dict[str, LoomActorGroup]":
        '''
        Setup model, optimizer, scheduler by different strategy
        This Function can either be implemented manually, or be replaced by train_strategy'''
        raise NotImplementedError
    
    def backward(self, actor, loss):
        '''
        This Function should implements the backward process.
        It can either be implemented manually, or be replaced by train_strategy'''
        raise NotImplementedError

    def step(self):
        '''
        This Function should implements the optimizer step process.
        It can either be implemented manually, or be replaced by train_strategy'''
        raise NotImplementedError

    def zero_grad(self):        
        '''
        This Function should implements the optimizer/model zero_grad process.
        It can either be implemented manually, or be replaced by train_strategy'''
        raise NotImplementedError

    def micro_batch_forward_backward(self, batch) -> "dict[str, Accumulator]":
        '''You May implement this function, or implement `forward_backward` directly.'''
        raise NotImplementedError

    def non_accum_logs_after_one_step(self) -> "dict[str, Accumulator]":
        return dict()

    def forward_backward(self, batches) -> "dict[str, Accumulator]":
        '''
        This Function defines a global step forward and backward process, aimed to gain the full grad of this batches. The optimizer step will be executed immediately after this function having been executed.

        [Note] batches is a global batch, a list of micro batch
        '''

        logs_dict = defaultdict(Accumulator)
        for batch in batches:
            mirco_logs_dict = self.micro_batch_forward_backward(batch)
            for k, v in mirco_logs_dict.items():
                if not isinstance(v, Accumulator): v = Accumulator(v, 1)
                logs_dict[k] += v
        return {k: v.get_value() for k, v in logs_dict.items()}


    def micro_batch_validate_forward(self, batch) -> "dict[str, Accumulator]":
        raise NotImplementedError


    def validate(self, val_data_iter):
        '''You may implement validating process in this function  and return a result dicts for visualization'''

        logs_dict = defaultdict(Accumulator)
        
        step_bar = tqdm(
                range(len(self.eval_dataloader)),
                desc = f"Eval stage of steps {global_step}",
                disable = parallel.get_rank() != 0
            )
        for batches in val_data_iter:
            for batch in batches:
                mirco_logs_dict = self.validate_forward(batch)
                for k, v in mirco_logs_dict.items():
                    logs_dict[k] += v
            step_bar.update()
        logs_dict = {k: v.get_value() for k, v in logs_dict.items()}
        step_bar.set_postfix(logs_dict)
        return logs_dict


    def update(self, batches):
        '''logic that forward/backward a whole batch then update parameters'''
        train_logs_dict = self.forward_backward(batches)
        self.step(self.opt_groups)
        self.zero_grad(self.opt_groups)
        train_logs_dict.update(self.non_accum_logs_after_one_step())
        
        return train_logs_dict
        





