import torch
from torch import nn
from typing import Optional
from loomtrain.core.actor import (
    LoomActor
)
from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.core.modeling.customs.rm_modeling import train_forwards


def get_actor_cls(actor_type: Literal["causal", "classifier"] = "causal") -> "GPT" | "RM":
    if actor_type == "causal":
        return PackingGPT
    
    elif actor_type == "classifier":
        return PackingRM



class Actor(nn.Module):
    def __init__(self, model: "nn.Module"):
        super().__init__()
        self.model = model



class PackingGPT(Actor):
    def forward(self, 
                sequeces: "torch.LongTensor",
                seq_lens: "Optional[list[int]]" = None,
                attention_mask: "Optional[torch.BoolTensor]" = None):
        inputs = parallel.prepare_cp_input(sequeces = sequeces, 
        seq_lens = seq_lens, attention_mask = attention_mask)
        output = self.model(*inputs)

        output["logits"] = output["logits"].to(torch.float32)

        return output

class PackingRM(Actor):
    def __init__(self, model):
        model._org_forward = model.forward
        
        train_forward = getattr(train_forwards,  model.config.architectures[0],
                                self.train_forward)

        model.forward = partial(train_forward, model = model)

        super().__init__(model)



    def train_forward(self,
                      input_ids: torch.LongTensor = None, #packed_sequences
                      attention_mask: Optional[torch.BoolTensor] = None,
                      position_ids: Optional[torch.LongTensor]= None,                     
                      model = None,
                      ** kwargs,
                      ):

        output: BaseModelOutputWithPast = model.model(input_ids = input_ids,
                                                      attention_mask = attention_mask,
                                                      position_ids = position_ids)
        hidden_states = output.last_hidden_state
        logits = model.score(hidden_states)

        output = SequenceClassifierOutputWithPast(
            logits = logits
        )        

        return output



    def forward(self,
                sequences: torch.LongTensor, #packed_sequences
                seq_lens: Optional[list[int]] = None,
                attention_mask: Optional[torch.BoolTensor] = None,
                ):

        inputs = parallel.prepare_cp_input(sequeces = sequeces, 
        seq_lens = seq_lens, attention_mask = attention_mask)
        output = self.model(*inputs)

        output["logits"] = output["logits"].to(torch.float32)

        return output


