from __future__ import annotations
import os
from copy import deepcopy
from typing import Union, Literal, Callable, TYPE_CHECKING
from datetime import timedelta
from dataclasses import dataclass
import torch, transformers, deepspeed
import torch.distributed as dist
from torch import nn
import torch.utils.data as tud
from ring_flash_attn import substitute_hf_flash_attn
from transformers import PreTrainedTokenizer
# from deepspeed import PipelineEngine
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from loomtrain.utils.distributed_sampler import (
    DistributedSampler, DistributedBucketSampler
)
if TYPE_CHECKING:
    from loomtrain.trainer.base import TrainerConfig
    from loomtrain.utils.lora import LoRAConfig
from peft import get_peft_model_state_dict, get_peft_model, PeftModel
# from loomtrain.strategy.base import Strategy

from loomtrain.core.strategy import TrainStrategy
from loomtrain.core.parallel import parallel_state as parallel

from loomtrain.strategy.deepspeed.utils import *
from loomtrain.modeling.gpt import GPT
from loomtrain.modeling.rm import RM
from loomtrain.utils.init_hf import init_model
from loomtrain.utils.common import IO


from loomtrain.core.actor import LoomOptDict, LoomActorGroup


class OrdinaryStrategy(TrainStrategy):
    '''
    This Strategy has no optimizer/grad offload, which means it may take much memory
    '''
    def __init__(
            self,
            parallel_config: "parallel.ParallelConfig",
            full_determinism: bool = False,
            seed: int = 42,
    ):
        super().__init__(
            parallel_config = parallel_config,
            full_determinism = full_determinism,
            seed = seed
        )
        

    def init_distributed(self):
        parallel.init_distributed(backend = 'nccl')


    def loomModule_setup_module(self, opt_dicts: "dict[str, LoomOptDict]") -> "dict[str, LoomActorGroup]":
        raise NotADirectoryError("If you use OrdinaryStrategy, make sure pass the `actor_groups` args into LoomModule")

    def loomModule_backward(self, actor, loss):
        loss.backward()
    
    def loomModule_step(self):
        for group in self.opt_groups.values():
            group.actor.optimizer.step()
            group.actor.scheduler.step()

    def loomModule_zero_grad(self):
        for group in self.opt_groups.values():
            group.actor.model.zero_grad()

    def loomModule_save_ckpt(self, save_dir: str, tag: str):
        for name, group in self.opt_groups.items():
            group.model.save_checkpoint(save_dir = os.path.join(save_dir, name), 
                                        tag = tag,
                                        client_state = dict(),
                                        save_latest = True)


    def loomModule_load_ckpt(self, saved_dir: str, tag: str):
        for name, group in self.opt_groups.items():
            group.model.load_checkpoint(
                load_dir = saved_dir,
                tag = tag,
                load_module_strict = True,
                load_optimizer_states = True,
                load_lr_scheduler_states = True,
                load_module_only = False
            )
    
    def loomModule_save_module(self, save_dir: str):        
        for name, group in self.opt_groups.items():
            gathered_state_dict = dict()
            model_to_save = group.model

            csave_dir = os.path.join(save_dir, name)

            if dist.get_rank() == 0:
                state_dict = model_to_save.state_dict()
                for k, v in model_to_save.named_buffers():
                    if k in state_dict:
                        gathered_state_dict[k] = v.data.cpu()
                
                state_dict_keys = set(state_dict.keys())
                gathered_state_dict_keys = set(gathered_state_dict.keys())

                assert state_dict_keys.issubset(gathered_state_dict), \
                f"Mismatch keys: {gathered_state_dict_keys.symmetric_difference(state_dict_keys)}"

                if isinstance(model_to_save, PeftModel):
                    if isinstance(model, GPT):
                        model_to_save = deepcopy(model_to_save)
                    elif isinstance(model, RM):
                        base_model = init_model(model_to_save.base_model.model._load_path,
                                                model_type = "classifier")
                        cloned = get_peft_model(base_model, self.lora_config)
                        cloned.load_state_dict(model_to_save.state_dict(), strict=True)
                        model_to_save = cloned


                    if self.lora_config.save_merged:
                        model_to_save = model_to_save.merge_and_unload()
                        model_to_save.save_pretrained(csave_dir, ** kwargs)
                    else:
                        adapter_csave_dir = csave_dir
                        model_to_save.save_pretrained(adapter_csave_dir, ** kwargs)
                        if self.config.zero_stage == 3:
                            torch.save(
                                get_peft_model_state_dict(model_to_save, gathered_state_dict),
                                os.path.join(csave_dir, "adapter_model.bin"),
                            )
                        
                else:
                    model_to_save.save_pretrained(
                        save_directory = csave_dir, state_dict = gathered_state_dict, **kwargs)

                model_to_save.config.to_json_file(os.path.join(csave_dir, "config.json"))

                tokenizer.save_pretrained(csave_dir)


                train_from_model_path = model_to_save.config._name_or_path
                if os.path.exists(train_from_model_path):
                    for file_name in IO.read_path(train_from_model_path, concat_root = False):
                        if file_name.endswith(".py"):
                            IO.copy(os.path.join(train_from_model_path, file_name),
                                    os.path.join(csave_dir, file_name))

            dist.barrier()
            torch.cuda.synchronize()