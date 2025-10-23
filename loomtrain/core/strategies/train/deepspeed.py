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

@dataclass
class DeepspeedConfig:
    zero_stage: Literal[2, 3] = 2,
    enable_bf16     : bool = True,
    offload         : bool = False
    adam_offload    : bool = False
    ref_offload     : bool = False # reference model offload
    grad_clip       : float = 1. 
    zpg             : int = 1
    grad_accum_dtype: Literal["fp16", "bf16", "fp32"] = None
    overlap_comm    : bool = False
    load_univeral   : bool = False
    torch_compile   : bool = False #useless ?


@dataclass
class RingAttnConfig:
    ring_attn_size  : int = 1
    ring_head_stride: int = 1

class DeepspeedStrategy(TrainStrategy):
    def __init__(
            self,
            parallel_config: "parallel.ParallelConfig",
            deepspeed_config: "DeepspeedConfig" = None,
            full_determinism: bool = False,
            seed: int = 42,
    ):
        super().__init__(
            parallel_config = parallel_config,
            full_determinism = full_determinism,
            seed = seed
        )
        self.config = deepspeed_config

    def init_distributed(self):
        deepspeed.init_distributed(timeout = self.init_timeout)


    def loomModule_setup_module(self, opt_dicts: "dict[str, LoomOptDict]") -> "dict[str, LoomActorGroup]":
        '''
        deepspeed default config only one model, one optimizer and one scheduler,
        for more flexible use, one may implement `setup_module` directly by inheriting LoomModule and override it.
        '''

        built_dict = dict()

        for name, actor_dict in actor_dicts.items():
            model = init_model(actor_dict.model_name, model_type = actor_dict.model_type)
            tokenizer = init_tokenizer(actor_dict.tokenizer_name)
            AdamOptimizer = DeepSpeedCPUAdam if self.config.adam_offload else FusedAdam
            optim_params = optimizer_grouped_parameters(model, actor_dict.weight_decay)
            optimizer = AdamOptimizer(optim_params, **actor_dict.pop('model'))

            scheduler = get_scheduler(
                name = actor_dict.lr_type,
                optimizer = optimizer,
                num_warmup_steps = actor_dict.num_warmup_steps,
                num_training_steps = actor_dict.total_steps,
                scheduler_specific_kwargs = dict(min_lr = actor_dict.lr * 0.1 \
                    if actor_dict.min_lr is None else actor_dict.min_lr)
            )
            model, optimizer, scheduler = self._prepare_train(
                model, optimizer, scheduler
            )

            built_dict[name] = LoomActorGroup(
                model = model,
                tokenizer = tokenizer,
                optimizer = optimizer,
                scheduler = scheduler,
                actor_type = actor_dict.actor_type,
                loss_type = actor_dict.loss_type
            )

        return built_dict

    def loomModule_backward(self, actor, loss):
        actor.model.backward(loss)
    
    def loomModule_step(self):
        for group in self.opt_groups.values():
            group.actor.model.step()

    def loomModule_zero_grad(self):
        for group in self.opt_groups.values():
            engine = group.actor.model
            if engine.bfloat16_enabled():
                # TODO: Temporary until bf16_optimizer and zero_optimizer are integrated
                if engine.zero_optimization() and hasattr(engine.optimizer, "zero_grad"):
                    engine.optimizer.zero_grad()
                else:
                    pass
            elif engine.zero_optimization() or engine.fp16_enabled() or engine.amp_enabled():
                engine.optimizer.zero_grad()
            else:
                engine.zero_grad()

    def _prepare_train(self, model: "nn.Module", optimizer, scheduler):
        engine, optimizer, _, scheduler = deepspeed.initialize(
            model = model,
            optimizer = optimizer,
            lr_scheduler = scheduler,
            config = deepspeed_train_config(
                        offload = self.config.offload,
                        adam_offload = self.config.adam_offload,
                        stage = self.config.zero_stage,
                        enable_bf16 = self.config.enable_bf16,
                        # train_batch_size = self.config.train_batch_size,
                        gradient_accumulation_steps = self.accumulated_gradient,
                        train_micro_batch_size_per_gpu = self.config.train_micro_batch_size_per_gpu,
                        grad_clip = self.config.grad_clip,
                        zpg = self.config.zpg,
                        grad_accum_dtype = self.config.grad_accum_dtype,
                        overlap_comm = self.config.overlap_comm,
                        load_univeral = self.config.load_univeral
                    ),
            args = dict(local_rank = int(os.environ.get("LOCAL_RANK", "-1"))),
            dist_init_required = True,
        ) 

        if self.config.torch_compile: engine.compile()

        return model, optimizer, scheduler
    

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
    

    # def restore_ckpt(self,
    #                  states: dict,
    #                  train_dataloader: tud.DataLoader,
    #                  config: TrainerConfig):
    #     consumed_samples = states['consumed_samples']
    #     total_tokens = states["total_tokens"] / self.ring_groups * (10**9)
    #     loss_tokens = states["loss_tokens"] / self.ring_groups * (10**9)
    #     update_steps_per_epoch = config.update_steps_per_epoch(train_dataloader)

    #     step = consumed_samples // config.batch_size * self.accumulated_gradient + 1
    #     start_epoch = consumed_samples // config.batch_size // update_steps_per_epoch
    #     consumed_samples %= (update_steps_per_epoch * config.batch_size)

    #     return step, update_steps_per_epoch, start_epoch, consumed_samples, total_tokens, loss_tokens
    
    def loomModule_save_module(self, save_dir: str):
        
        for name, group in self.opt_groups.items():
            gathered_state_dict = dict()
            model_to_save = group.model

            csave_dir = os.path.join(save_dir, name)

            for k, v in model_to_save.named_parameters():

                with deepspeed.zero.GatheredParameters([v], enabled = \
                                                    hasattr(v, "ds_id") and v.ds_status == ZeroParamStatus.NOT_AVAILABLE):
                    if dist.get_rank() == 0:
                        gathered_state_dict[k] = v.data.cpu()
            
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