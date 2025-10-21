from typing import Literal, List
from dataclasses import dataclass
import torch.distributed as dist
from loomtrain.core.device.mesh import DeviceMesh

@dataclass
class ParallelConfig:
    train_batch_size: int = 1
    micro_batch_size: int = 1
    val_batch_size: int = 1
    nnodes: int = 1
    devices_per_node: int = 8
    cp:int = 8
    pp:int = 1
    sp:int = 1
    tp:int = 1
    order = ("tp", "sp", "cp", "pp", "dp")

    @property
    def grad_accum(self):
        return self.train_batch_size * self.cp // self.micro_batch_size // self.expected_world_size

    @property
    def shape(self):
        return tuple([getattr(self, o) for o in self.order])

    @property
    def expected_order(self):
        return self.order

    @property
    def expected_world_size(self):
        return self.nnodes * self.devices_per_node

    @property
    def size_expect_dp(self):
        return self.cp * self.pp * self.sp * self.tp
    
    @property
    def dp(self):
        if not hasattr(self, "_dp"):
            self._dp = self.expected_world_size // self.size_expect_dp
        return self._dp


    def __post_init__(self):
        assert len(self.order) == 5, str(self.order)
        for p in ("cp", "dp", "pp", "sp", "tp"):
            assert p in self.order, str(self.order)
        

        assert self.expected_world_size % self.size_expect_dp == 0, \
            f"The Nodes({self.nnodes}) * Devices({self.devices_per_node}) you provide doesn't match the parallel config:({self.shape})"



_CP_GROUP_: dist.ProcessGroup = None
_DP_GROUP_: dist.ProcessGroup = None
_SP_GROUP_: dist.ProcessGroup = None
_TP_GROUP_: dist.ProcessGroup = None
_PP_GROUP_: dist.ProcessGroup = None

_CP_RANKS_: list = None
_DP_RANKS_: list = None
_SP_RANKS_: list = None
_TP_RANKS_: list = None
_PP_RANKS_: list = None

_CP_RANK_: int = None
_DP_RANK_: int = None
_SP_RANK_: int = None
_TP_RANK_: int = None
_PP_RANK_: int = None

_CP_WORLD_SIZE_: int = None
_DP_WORLD_SIZE_: int = None
_SP_WORLD_SIZE_: int = None
_TP_WORLD_SIZE_: int = None
_PP_WORLD_SIZE_: int = None

_RANK_: int = None
_WORLD_SIZE_: int = None

_IS_INITIALIZED_: bool = False


def get_cp_group():
    assert _CP_GROUP_ is not None
    return _CP_GROUP_

def get_dp_group():
    assert _DP_GROUP_ is not None
    return _DP_GROUP_

def get_sp_group():
    assert _SP_GROUP_ is not None
    return _SP_GROUP_

def get_tp_group():
    assert _TP_GROUP_ is not None
    return _TP_GROUP_

def get_pp_group():
    assert _PP_GROUP_ is not None
    return _PP_GROUP_

def set_cp_group(group: dist.ProcessGroup):
    global _CP_GROUP_
    _CP_GROUP_ = group

def set_dp_group(group: dist.ProcessGroup):
    global _DP_GROUP_
    _DP_GROUP_ = group

def set_sp_group(group: dist.ProcessGroup):
    global _SP_GROUP_
    _SP_GROUP_ = group

def set_tp_group(group: dist.ProcessGroup):
    global _TP_GROUP_
    _TP_GROUP_ = group

def set_pp_group(group: dist.ProcessGroup):
    global _PP_GROUP_
    _PP_GROUP_ = group


def get_cp_ranks():
    assert _CP_RANKS_ is not None
    return _CP_RANKS_
    
def get_dp_ranks():
    assert _DP_RANKS_ is not None
    return _DP_RANKS_
    
def get_sp_ranks():
    assert _SP_RANKS_ is not None
    return _SP_RANKS_
    
def get_tp_ranks():
    assert _TP_RANKS_ is not None
    return _TP_RANKS_
    
def get_pp_ranks():
    assert _PP_RANKS_ is not None
    return _PP_RANKS_


def set_cp_ranks(ranks: List[int]):
    global _CP_RANKS_ 
    _CP_RANKS_ = ranks

def set_dp_ranks(ranks: List[int]):
    global _DP_RANKS_ 
    _DP_RANKS_ = ranks

def set_sp_ranks(ranks: List[int]):
    global _SP_RANKS_ 
    _SP_RANKS_ = ranks

def set_tp_ranks(ranks: List[int]):
    global _TP_RANKS_ 
    _TP_RANKS_ = ranks

def set_pp_ranks(ranks: List[int]):
    global _PP_RANKS_ 
    _PP_RANKS_ = ranks


def get_cp_rank():
    assert _CP_RANK_ is not None
    return _CP_RANK_
    
def get_dp_rank():
    assert _DP_RANK_ is not None
    return _DP_RANK_
    
def get_sp_rank():
    assert _SP_RANK_ is not None
    return _SP_RANK_
    
def get_tp_rank():
    assert _TP_RANK_ is not None
    return _TP_RANK_
    
def get_pp_rank():
    assert _PP_RANK_ is not None
    return _PP_RANK_


def set_cp_rank(rank: int):
    global _CP_RANK_ 
    _CP_RANK_ = rank

def set_dp_rank(rank: int):
    global _DP_RANK_ 
    _DP_RANK_ = rank

def set_sp_rank(rank: int):
    global _SP_RANK_ 
    _SP_RANK_ = rank

def set_tp_rank(rank: int):
    global _TP_RANK_ 
    _TP_RANK_ = rank

def set_pp_rank(rank: int):
    global _PP_RANK_ 
    _PP_RANK_ = rank


def get_cp_size():
    assert _CP_WORLD_SIZE_ is not None
    return _CP_WORLD_SIZE_
    
def get_dp_size():
    assert _DP_WORLD_SIZE_ is not None
    return _DP_WORLD_SIZE_
    
def get_sp_size():
    assert _SP_WORLD_SIZE_ is not None
    return _SP_WORLD_SIZE_
    
def get_tp_size():
    assert _TP_WORLD_SIZE_ is not None
    return _TP_WORLD_SIZE_
    
def get_pp_size():
    assert _PP_WORLD_SIZE_ is not None
    return _PP_WORLD_SIZE_


def set_cp_size(size: int):
    global _CP_WORLD_SIZE_ 
    _CP_WORLD_SIZE_ = size

def set_dp_size(size: int):
    global _DP_WORLD_SIZE_ 
    _DP_WORLD_SIZE_ = size

def set_sp_size(size: int):
    global _SP_WORLD_SIZE_ 
    _SP_WORLD_SIZE_ = size

def set_tp_rank(size: int):
    global _TP_WORLD_SIZE_ 
    _TP_WORLD_SIZE_ = size

def set_pp_rank(size: int):
    global _PP_WORLD_SIZE_ 
    _PP_WORLD_SIZE_ = size



def get_cp_count():
    return get_world_size() // get_cp_size()
    
def get_dp_count():
    return get_world_size() // get_dp_size()
    
def get_sp_count():
    return get_world_size() // get_sp_size()
    
def get_tp_count():
    return get_world_size() // get_tp_size()
    
def get_pp_count():
    return get_world_size() // get_pp_size()



def get_parallel_group(type: Literal["tp", "sp", "cp", "pp", "dp"]):
    return globals()[f"get_{type}_group"]()

def set_parallel_group(group: dist.ProcessGroup, type: Literal["tp", "sp", "cp", "pp", "dp"]):
    return globals()[f"set_{type}_group"](group)    


def get_parallel_ranks(type: Literal["tp", "sp", "cp", "pp", "dp"]):
    return globals()[f"get_{type}_ranks"]()

def set_parallel_ranks(ranks: List[int], type: Literal["tp", "sp", "cp", "pp", "dp"]):
    return globals()[f"set_{type}_ranks"](ranks)    


def set_parallel_rank(rank: int, type: Literal["tp", "sp", "cp", "pp", "dp"]):
    return globals()[f"set_{type}_rank"](rank)


def set_parallel_size(size: int, type: Literal["tp", "sp", "cp", "pp", "dp"]):
    return globals()[f"set_{type}_size"](size)    

def get_rank():
    assert _RANK_ is not None
    return _RANK_

def is_rank0():
    assert _RANK_ is not None
    return _RANK_ == 0

def set_rank(rank: int = None):
    global _RANK_
    if rank is None: rank = dist.get_rank()
    _RANK_ = rank

def get_world_size():
    assert _WORLD_SIZE_ is not None
    return _WORLD_SIZE_

def set_world_size(size: int = None):
    global _WORLD_SIZE_
    if size is None: size = dist.get_world_size()
    _WORLD_SIZE_ = size



def set_initialized():
    global _IS_INITIALIZED_
    _IS_INITIALIZED_ = True

def init_parallel_groups(parallel_config: "ParallelConfig"):
    device_mesh = DeviceMesh(parallel_config)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    for parallel_type in parallel_config.order:
        parallel_group_ranks = device_mesh[parallel_type]
        assert rank in parallel_group_ranks \
            and len(parallel_group_ranks) == getattr(device_mesh.parallel_config, parallel_type), f"Group:{parallel_group_ranks}, Rank:{rank}, Parallel: {parallel_type} = {getattr(device_mesh.parallel_config, parallel_type)}"
        
        #TODO: the argument 'backend' should be configurable
        group = dist.new_group(ranks = parallel_group_ranks, backend = "nccl")

        set_parallel_group(group, parallel_type)
        set_parallel_ranks(parallel_group_ranks, parallel_type)
        set_parallel_size(len(parallel_group_ranks), parallel_type)
        set_parallel_rank(parallel_group_ranks.index(rank), parallel_type)
        set_rank(rank)
        set_world_size(world_size)
    
    set_initialized()

def is_initialized() -> bool:
    return _IS_INITIALIZED_


class ParallelPlugin:
    '''
    For different specific parallel method to inherit.
    '''
    def initialize(self):
        raise NotImplementedError



def prepare_cp_input(*args, **kwargs):
    raise NotImplementedError
    




