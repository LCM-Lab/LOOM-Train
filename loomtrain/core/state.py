from dataclasses import dataclass
import torch.distributed as dist
from loomtrain.utils.common.iotools import IO
from loomtrain.core.strategy import *


@dataclass
class CheckpointConfig:
    load_dir: str
    save_dir: str
    
    do_resume: bool = True
    ckpt_interval: int = 10
    weight_interval: int = 10
    visulization_interval: int = 10
    max_ckpts: int = 2
    max_ckpts_GB: int = 1024


class CheckpointMixin:
    '''
    This class automatically save training status
    '''
    def __init__(self):
        self._global_step = 0
    
    @property
    def global_step(self): return self._global_step


    @property
    def state(self):
        '''The state to save'''
        raise NotImplementedError

    def get_saved_sub_dir(self) -> str:
        ''' sub_dir mainly for different types or checkpoint '''
        raise NotImplementedError
    

    def update(self):
        '''update the state'''
        raise NotImplementedError

    def save_ckpt(self, save_dir: str, tag: str):
        raise NotImplementedError
    
    def load_ckpt(self, saved_dir: str, tag: str):
        raise NotImplementedError
    
    def _get_saving_interval(self, checkpoint_config: "CheckpointConfig") -> bool:
        '''extract saving interval from checkpoint_config and return it'''
        return checkpoint_config.ckpt_interval

    def _update(self, *args, **kwargs):
        self._global_step += 1
        return self.update(*args, **kwargs)
        

    def _save_ckpt(self, checkpoint_config: "CheckpointConfig", inplace: bool = False):
        if self._get_saving_interval(checkpoint_config) % self.global_step: return
        save_dir = os.path.join(checkpoint_config.save_dir, self.get_saved_sub_dir())

        tag = f"global_step{self.global_step}"
        max_ckpts = checkpoint_config.max_ckpts
        max_ckpt_GB = checkpoint_config.max_ckpts_GB


        
        if dist.get_rank() == 0:
            os.makedirs(save_dir, exist_ok = True)
        dist.barrier()

        if not inplace: # None means no need to save seperately
            if dist.get_rank() == 0:
                MAX_SIZE = max_ckpt_GB * 1024**3
                subdirs = sorted([k for k in IO.read_path(save_dir) if os.path.isdir(k)],
                                key = lambda x: os.path.getmtime(x))
                
                while True:
                    total_size = sum(
                        os.path.getsize(os.path.join(dir_path, file_name))
                        for subdir in subdirs
                        for dir_path, folder_names, file_names in os.walk(subdir)
                        for file_name in file_names
                    )

                    if len(subdirs) < max_ckpts and total_size <= MAX_SIZE:
                        break

                    IO.remove(subdirs.pop(0))

            dist.barrier()

        self.save_ckpt(save_dir, tag = tag)
        
        if not inplace:
            with open(os.path.join(save_dir, "latest"), "w") as f:
                f.write(tag)    

        if dist.get_rank() == 0:
            print(f"{self.__class__.__name__} Checkpoint: {save_dir}/{tag} is ready !!!")

    

    def _load_ckpt(self, checkpoint_config: "CheckpointConfig"):
        saved_dir = checkpoint_config.save_dir
        saved_dir = os.path.join(saved_dir, self.get_saved_sub_dir())
        latest_path = os.path.join(saved_dir, "latest")
        if os.path.exists(latest_path): # only for those ckpts having multiple files
            with open(latest_path, "r") as f:
                tag = f.read().strip()
        try:
            return self.load_ckpt(saved_dir, tag)    
        finally:
            if dist.get_rank() == 0:
                print(f"Successfully load {self.__class__.__name__} Checkpoint from: {saved_dir} !!!")
