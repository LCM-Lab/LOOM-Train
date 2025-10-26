import torch
import torch.utils.data as tud
from typing import Literal

def role_template(message: str | list[dict[str, str]], role: Literal["system", "user", "assistant"]):
    if isinstance(message, str):
        message = [{"role": role, "content": message}]
    return message

class CollateDataset(tud.Dataset): # TODO: add tokenizer 
    def collate_fn(self,item_list):
        return torch.stack(item_list)
    def initialize(self, * args, **kwargs):
        '''
        The initialization of this kind of dataset is not occurred when be instantiated, 
        but when dataloader is created. So that different rank can initialize different part
        of the dataset, which is more efficient.
        '''
        raise NotImplementedError
