import torch
import torch.distributed as dist
import torch.optim as opt
import torch.utils.data as tud
import torch.optim.lr_scheduler as tol
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
# from einops import rearrange,reduce,repeat
# from einops.layers.torch import Rearrange,Reduce
from transformers import PreTrainedTokenizer
from loomtrain.utils.distributed.torch import all_reduce
from loomtrain.utils.wandb import WandbConfig
from loomtrain.trainer.base import Trainer, TrainerConfig
from loomtrain.modeling.gpt import GPT, GPTCELoss
from loomtrain.strategy import DeepspeedStrategy
from loomtrain.dataset.sft import SFTDataset
from loomtrain.utils.distributed_sampler import DistributedSampler

  
class LMTrainer(Trainer):
    '''
    strategy will wrap the model and setup dataloader here
    '''
    def __init__(
      self,
      model: GPT,
      train_dataset: SFTDataset,
      eval_dataset: SFTDataset,
      optimizer: opt.Optimizer,
      strategy: DeepspeedStrategy,
      config: TrainerConfig,
      tokenizer: PreTrainedTokenizer = None,
      save_hf_ckpt: bool = False,
      disable_ds_ckpt: bool = False,
      wandb_config: WandbConfig = None,     
    ):
        super().__init__(
            model = model,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            optimizer = optimizer,
            strategy = strategy,
            config = config,
            wandb_config = wandb_config
        )

        self.tokenzier = tokenizer
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt


        self.loss_fn = GPTCELoss(ring_attn_group = strategy.ring_attn_group)


    def fit(self, load_ckpt: bool = True):
        states = self.load_ckpt(load_ckpt = load_ckpt)
        consumed_samples = states['consumed_samples']
        update_steps_per_epoch = self.config.update_steps_per_epoch(len(self.train_dataloader.dataset))
        
        assert len(self.train_dataloader.dataset) and update_steps_per_epoch, \
            f"train_dataset_len: {len(self.train_dataloader.dataset)} < batch_size: {self.config.batch_size}"
        if self.config.eval_steps < 0:
            self.config.eval_steps = update_steps_per_epoch
        

        step = consumed_samples // self.config.batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // self.config.batch_size // update_steps_per_epoch
        consumed_samples %= (update_steps_per_epoch * self.config.batch_size)

        epoch_bar = tqdm(range(start_epoch, self.config.max_epochs), 
                         desc = "Train epoch", 
                         disable = dist.get_rank() != 0)


        # scheduler_steps_per_epoch = len(self.train_dataloader.dataset) // len(self.train_dataloader.batch_size)


        loss = 0
        for epoch in range(start_epoch, self.config.max_epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples = 0 if epoch > start_epoch else consumed_samples 
                )
            
            step_bar = tqdm(range(update_steps_per_epoch),
                            desc = f"Train step of epoch {epoch}",
                            disable = dist.get_rank() != 0)
            
            self.model.train()
            for prompt_id_lens, inputs, attention_masks, infos in self.train_dataloader:
                inputs = inputs.to(torch.cuda.current_device())                
                attention_mask = attention_masks.to(torch.cuda.current_device())

                output =  self.model(
                    sequences = inputs,
                    seq_lens = infos["input_length"],
                    attention_mask = attention_mask,
                    ring_attn_group = self.strategy.ring_attn_group
                )
                

                labels = torch.where(attention_mask.bool(), 
                                     inputs, 
                                     self.loss_fn.ignore_index)

                index = 0
                for input_length, prompt_id_len in zip(infos["input_length"], prompt_id_lens):
                    labels[0][index: index + prompt_id_len + 1] = self.loss_fn.ignore_index
                    index += input_length
            
                gpt_loss = self.loss_fn(output.logits, labels)

                self.strategy.backward(gpt_loss, self.model)
                self.strategy.optimizer_step(self.model)

                loss += gpt_loss.item()

                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict = dict(
                        gpt_loss = gpt_loss.item(),
                        lr = self.scheduler.get_last_lr()[0]
                    )
                    logs_dict = all_reduce(logs_dict)

                    logs_dict["mean_loss"] = loss / self.strategy.accumulated_gradient
                    loss = 0
                    
                    step_bar.set_postfix(logs_dict)
                    step_bar.update()
                    
                    global_step = step // self.strategy.accumulated_gradient

                    self.update_wandb({f"train/{k}": v for k, v in \
                                       {**logs_dict, "global_step": global_step}.items()}, 
                                       global_step, 
                                       self.config.logging_steps,
                                       step = global_step)
                    self.evaluate(global_step)
                    self.save_ckpt(global_step = global_step, 
                                   client_state = dict(consumed_samples = global_step * self.config.batch_size))

                step += 1
            epoch_bar.update()
    
        self.finish_wandb()


    def evaluate(self, global_step:int = 0):
        if global_step % self.config.eval_steps: return
        self.model.eval()
        with torch.no_grad():
            loss = 0
            step_bar = tqdm(
                range(len(self.eval_dataloader)),
                desc = f"Eval stage of steps {global_step}",
                disable = dist.get_rank() == 0,
            )
            for times, (prompt_id_lens, inputs, attention_masks, infos) in enumerate(self.eval_dataloader):
                inputs = inputs.to(torch.cuda.current_device())
                attention_mask = attention_masks.to(torch.cuda.current_device())
                output = self.model(sequences = inputs, 
                                    seq_lens = infos["input_length"],
                                    attention_mask=attention_mask,
                                    ring_attn_group = self.strategy.ring_attn_group)
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.ignore_index
                )

                index = 0
                for input_length, prompt_id_len in zip(infos["input_length"], prompt_id_lens):
                    labels[0][index: index + prompt_id_len] = self.loss_fn.ignore_index
                    index += input_length
                
                gpt_loss = self.loss_fn(output.logits, labels)

                loss += gpt_loss.item()

                bar_dict = {"eval gpt_loss": loss / (times + 1)}

                step_bar.update()
                logs = all_reduce(bar_dict)
                step_bar.set_postfix(logs)
            
            self.update_wandb({f"eval/{k}": v for k, v in {**logs, "global_step": global_step}.items()},
                              global_step, self.config.logging_steps, step = global_step)
        
        self.model.train()