import torch
from loomtrain.core.actor import LoomOptDict
from loomtrain.core.module import LoomModule
from loomtrain.core.datamodule import LoomDataModule
from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.utils.distributed.torch import all_reduce


class LoomSFT(LoomModule):
    def __init__(self, model_name: str):
        opt_dicts = dict(
            group0 = LoomOptDict(
                model_name = model_name,
                model_type = 'causal',
                loss_type = "sft"
            )
        )

        super().__init__(opt_dicts)
    
    def setup_self_module(self):
        self.actor = self.opt_groups['group0'].actor
        self.toknizer = self.opt_groups['group0'].tokenizer
        self.optimizer = self.opt_groups['group0'].optimizer
        self.scheduler = self.opt_groups['group0'].scheduler, 
        self.loss_fn = self.opt_groups['group0'].loss_fn

    def micro_batch_forward_backward(self, batch) -> "dict[str, obj]":
        inputs, attention_masks, loss_masks, seq_lens = batch
        outupt = self.actor(sequences = inputs, attention_masks = attention_masks,seq_lens = seq_lens)
        labels = torch.where(attention_mask.bool() & loss_masks.bool(), inputs, self.loss_fn.ignore_index)

        gpt_loss = self.loss_fn(output.logits, labels)

        self.backward(self.actor, gpt_loss)

        return dict(
            loss = gpt_loss.item(),
            total_tokens = all_reduce(sum(seq_lens)) * parallel.get_dp_count() / 10 ** 9,
            loss_tokens = all_reduce(loss_masks.int().sum().item()) * parallel.get_dp_count() / 10 ** 9
        )

    def micro_batch_validate_forward(self, batch):
        inputs, attention_masks, loss_masks, seq_lens = batch
        outupt = self.actor(sequences = inputs, attention_masks = attention_masks,seq_lens = seq_lens)
        labels = torch.where(attention_mask.bool() & loss_masks.bool(), inputs, self.loss_fn.ignore_index)

        gpt_loss = self.loss_fn(output.logits, labels)

        return dict(
            loss = gpt_loss.item(),
            total_tokens = all_reduce(sum(seq_lens)) * parallel.get_dp_count() / 10 ** 9,
            loss_tokens = all_reduce(loss_masks.int().sum().item()) * parallel.get_dp_count() / 10 ** 9
        )
    
    def non_accum_logs_after_one_step(self):
        return dict(lr = self.scheduler.get_last_lr()[0])