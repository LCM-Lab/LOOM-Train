from typing import List
import torch
import torch.nn.functional as F
import torch.distributed as dist 
from functools import partial
from dataclasses import dataclass
from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.core.parallel.context_parallel.base import ContextParallel

from ring_flash_attn import update_ring_flash_attn_params

def get_local_position_ids(start: int, end: int, seq_lens: List[int]):

    position_ids = torch.zeros((1, end - start), 
                               dtype = torch.long, 
                               device = torch.cuda.current_device())
    cumsum = 0
    for seq_len in seq_lens:
        seq_start = max(cumsum, start)
        seq_end = min(cumsum + seq_len, end)
        if seq_start < seq_end:
            position_ids[0, seq_start - start: seq_end - start] = \
                torch.arange(seq_start - cumsum, seq_end - cumsum)
        cumsum += seq_len
        if cumsum >= end:
            break
    return position_ids


def set_cu_seqlens_for_ring_attn(
        packed_sequences: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        seq_lens: List[int],
        ring_attn_group: dist.ProcessGroup = None
):
    '''
    sequences:(seq_len, ) 
    '''
    ring_attn_rank = dist.get_rank(group = ring_attn_group)
    ring_attn_size = dist.get_world_size(group = ring_attn_group)
    total_seq_len = packed_sequences.size(0) * packed_sequences.size(1)
    assert total_seq_len % ring_attn_size == 0, f"Packed Sequences shape: {packed_sequences.size()}"
    local_seq_len = total_seq_len // ring_attn_size
    start, end = ring_attn_rank * local_seq_len, (ring_attn_rank + 1) * local_seq_len
    sequences = packed_sequences[:, start: end]
    attention_mask = attention_mask[:, start: end]

    position_ids = get_local_position_ids(start, end, seq_lens)


    cu_seqlens = torch.cumsum(
        torch.tensor(seq_lens, device = torch.cuda.current_device(), dtype = torch.int32),
        dim = -1, dtype = torch.int32
    )

    update_ring_flash_attn_params(
        F.pad(F.pad(cu_seqlens, (1, 0), value=0), (0, 1), value=total_seq_len),
        ring_attn_group
    )

    return dict(sequences = sequences, 
                attention_mask = attention_mask, 
                position_ids = position_ids)




@dataclass
class RingAttnConfig:
    ring_type: str = "llama"
    ring_head_stride: int = 1


class RingFlashAttnPlugin(parallel.ParallelPlugin, ContextParallel):
    def __init__(self, config: "RingAttnConfig"):
        self.config = config
    
    def initialize(self):
        from ring_flash_attn import substitute_hf_flash_attn
        ring_attn_group = parallel.get_cp_group()
        substitute_hf_flash_attn(ring_attn_group, 
                                 self.config.ring_head_stride)
        
        parallel.prepare_cp_input = partial(
            substitute_hf_flash_attn, ring_attn_group = ring_attn_group
        )


