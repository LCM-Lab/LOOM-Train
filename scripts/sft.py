import sys, pickle, os
os.environ["CUDA_LAUNCH_BLOCKING"] ="1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

from datasets import load_from_disk
from loomtrain.utils.common.args import *
from loomtrain.strategy import (
    DeepspeedConfig, 
    RingAttnConfig, 
    DeepspeedStrategy
)
from loomtrain import cached, basename
from loomtrain.utils.wandb import WandbConfig
from loomtrain.utils.tensorboard import TensorboardConfig
from loomtrain.trainer import TrainerConfig, SFTTrainer
from loomtrain.modeling import GPT
from loomtrain.dataset.blended import BlendedSFTDataset
from loomtrain.dataset.sft import SFTDataset
from loomtrain.utils.init_hf import init_model, init_tokenizer
import torch.distributed as dist

def train(args):
    trainer_config = TrainerConfig(
        batch_size = args.batch_size,
        bucket_size = args.bucket_size,
        micro_batch_size = args.micro_batch_size,
        max_epochs = args.max_epochs,
        scheduler = "cosine_with_min_lr",
        learing_rate = getattr(args, "learning_rate", 5e-6),
        lr_warmup_ratio = getattr(args, "lr_warmup_ratio", 0.03),
        save_steps = getattr(args, "save_steps", -1),
        ckpt_dir = getattr(args,"ckpt_dir", None),
        max_ckpts = getattr(args,"max_ckpts", 1),
        weights_dir = getattr(args, "weights_dir", None),
        eval_steps = getattr(args, "eval_steps", 20),
        max_weights = args.max_weights,
        weights_saving_interval = args.weights_saving_interval,
    )

    deepspeed_config = DeepspeedConfig(
        zero_stage = getattr(args, "zero_stage", 2),
        enable_bf16 = getattr(args, "enable_bf16", False),
        offload = getattr(args, "offload", False),
        adam_offload = getattr(args, "adam_offload", False),
        train_batch_size = trainer_config.batch_size,
        train_micro_batch_size_per_gpu = trainer_config.micro_batch_size,
        grad_clip = getattr(args, "grad_clip", 1.0),
        zpg = getattr(args,"zpg", 1),
    )

    ring_attn_config = RingAttnConfig(
        ring_attn_size = getattr(args,"ring_attn_size", 8),
        ring_head_stride = getattr(args, "ring_head_stride", 1),    
    )


    # wandb_config = WandbConfig(
    #     api_key = args.wandb_api,
    #     entity = args.wandb_entity,
    #     project = args.wandb_project,
    #     name = args.wandb_name,
    #     group = args.wandb_group,
    #     config = dict(**vars(trainer_config))
    # )

    tensorboard_config = TensorboardConfig(
        log_dir = args.tensorboard_logdir,
        name = args.tensorboard_name
    )

    strategy = DeepspeedStrategy(
        seed = getattr(args, "seed", 42),
        deepspeed_config = deepspeed_config,
        ring_attn_config = ring_attn_config
    )

    strategy.init_distributed()

    if dist.get_rank() == 0:
        print("WORLD_SIE####     : ", dist.get_world_size())

    sft_model = init_model(getattr(args, "load_from", args.model_path))
    tokenizer =init_tokenizer(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = GPT(sft_model)

    datasets_dicts = [
        load_from_disk(pth) for pth in args.dataset_paths
    ]

    if dist.get_rank() == 0:
        [
            cached(
                SFTDataset,
                kwargs = dict(
                    dataset = dataset["train"],
                    prompt_key = "chat_template", response_key = "golden",
                    tokenizer = tokenizer, max_length = args.max_length,
                    ring_attn_size = ring_attn_config.ring_attn_size
                ),
                cache_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/dataset_files/" + \
                f"{basename(args.model_path)}={basename(data_pth)}=max-length128k"
            ) for dataset,data_pth in zip(datasets_dicts, args.dataset_paths)
        ]
        [
            cached(
                SFTDataset,
                kwargs = dict(
                    dataset = dataset["eval"],
                    prompt_key = "chat_template", response_key = "golden",
                    tokenizer = tokenizer, max_length = args.max_length,
                    ring_attn_size = ring_attn_config.ring_attn_size
                ),
                cache_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/dataset_files/" + \
                f"{basename(args.model_path)}={basename(data_pth)}=max-length128k-eval"
            ) for dataset,data_pth in zip(datasets_dicts, args.dataset_paths)
        ]

    dist.barrier()

    sft_datasets =  [
            cached(
                SFTDataset,
                kwargs = dict(
                    dataset = dataset["train"],
                    prompt_key = "chat_template", response_key = "golden",
                    tokenizer = tokenizer, max_length = args.max_length,
                    ring_attn_size = ring_attn_config.ring_attn_size
                ),
                cache_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/dataset_files/" + \
                f"{basename(args.model_path)}={basename(data_pth)}=max-length128k"
            ) for dataset,data_pth in zip(datasets_dicts, args.dataset_paths)
        ]

    sft_datasets_eval =  [
            cached(
                SFTDataset,
                kwargs = dict(
                    dataset = dataset["eval"],
                    prompt_key = "chat_template", response_key = "golden",
                    tokenizer = tokenizer, max_length = args.max_length,
                    ring_attn_size = ring_attn_config.ring_attn_size
                ),
                cache_path = f"{os.path.dirname((os.path.dirname(os.path.abspath(__file__))))}/dataset_files/" + \
                f"{basename(args.model_path)}={basename(data_pth)}=max-length128k-eval"
            ) for dataset,data_pth in zip(datasets_dicts, args.dataset_paths)
        ]
    for d in sft_datasets:
        d.ring_attn_size = args.ring_attn_size

    train_dataset = BlendedSFTDataset(sft_datasets,
                                      sample_ratios = args.sample_ratios,
                                      sample_counts = args.sample_counts)
    eval_dataset = BlendedSFTDataset(sft_datasets_eval, 
                                     sample_counts = args.sample_counts_eval)



    trainer = SFTTrainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        optimizer =  strategy.setup_optimizer(model, 
                                              lr = args.learning_rate, 
                                              betas = args.adam_betas, 
                                              weight_decay = args.L2),
        strategy = strategy,
        config = trainer_config,
        tokenizer = tokenizer,
        # wandb_config = wandb_config,
        tensorboard_config = tensorboard_config
    )

    trainer.fit(load_ckpt = args.do_resume) #看看要不要实时上传ckpt

    # strategy.save_model(model, tokenizer, args.save_dir)



# deepspeed --module train --deepspeed_sequence_parallel_size 8
# nohup deepspeed --module train_ywj > log 2>&1 &
# torchrun --nproc_per_node 8 train.py 

if __name__ == "__main__":
    workspace_path = f"{os.path.dirname(os.path.abspath(__file__))}/"
    save_name = "LR-Llama-3.1-8B-Instuct-formal-no-code-s1"
    args = AutoParser([
        StoreArgs(
            local_rank = -1,
            workspace_path = workspace_path,
            
            do_resume = True,
            dataset_paths = [
                "/data/jbb/datas/LongMiT-Reward-SFT-split_0-10-sampling10000",
                "/data/jbb/datas/LongMiT-Reward-Summ-SFT",
                "/data/jbb/datas/LongReward-Safety-SFT",
                "/data/jbb/datas/LongReward-Chat-SFT",
                # "/data/jbb/datas/LongReward-Code-SFT"
                
            ],
            sample_ratios = [1, #0.8,
                            #   0.1
                             ],
            sample_counts = [3000, 2000, 1500, 1200, 
                            #  500
                             ],
            sample_counts_eval = [50, 25, 20, 15, 
                                #   10
                                  ],
            model_path = "/data/hf_models/Meta-Llama-3.1-8B-Instruct",
            # load_from = "/data/jbb/long/train/s1_train/s1_train/weights-full-diverse/global_step30",
            weights_dir = f"{workspace_path}/{save_name}-weight",
            ckpt_dir = f"{workspace_path}/{save_name}-ckpt",
            tensorboard_name = save_name,
            save_steps = 15, 
            max_weights = 40,
            max_ckpts = 2,
            weights_saving_interval = 30,
            eval_steps = 10,
            #offload = True,
            #adam_offload = "cpu",
            max_length = 128000,
            bucket_size = 128000,

            zero_stage = 2,
            micro_batch_size = 1,
            batch_size = 16,
            ring_attn_size = 8,
            ring_head_stride = 4,

            max_norm = 1.0,
            seed = 42,
            enable_bf16 = True,
            zpg = 1,
            overlap_comm = True,
            auto_resume = False,
            max_epochs = 1,
            learning_rate = 2e-6,
            lr_warmup_ratio = 0.03,
            L2 = 0,
            adam_betas = (0.9, 0.95),

            # wandb_api = "e7aa2dce89a6c07877419927751e5ba432624aed",
            # wandb_entity = "iiiigray",
            # wandb_project = "LongReward",
            # wandb_name = "format-training", 
            # wandb_group = "first",
            tensorboard_logdir = f"{workspace_path}/tensorboard", ####
        )

    ]).parse_args()


    train(args)