import argparse
import builtins
from typing import Any, List, Literal


class Arg:
    def __init__(
            self, 
            name: str = None,
            short: str = None,
            type: type = None,
            action: Literal["store", "store_true", "store_false", "store_const", "count", "append"] = "store",
            default: Any = None,
            islist: bool = False,
            help: str = None,
            **kwargs
    ):        

        if name is None:
            assert len(kwargs) == 1
            name = next(iter(kwargs.keys()))
        
        if len(kwargs) == 1 and (action != "store_const"):
           assert name in kwargs, "You should either use `name` or default args! Not simultaneously!"
           default = kwargs[name]

        if type is None:
            if isinstance(default, builtins.type):
                type = default
                default = None
            else: type = builtins.type(default)

            

        self.list = [f"--{name.strip('_')}"]
        if short is not None:
            self.list += [f"-{short.strip('_')}"]
        assert type != None, "You passed a 'None' type and you didn't pass any default args"
        self.dict = dict()

        if action == "store":
            self.dict["type"] = type
        

        if default is not None:
            self.dict["default"] = type(default)

        if action != "store":
            self.dict["action"] =action

        if action == "store_const":
            dest, const = next(iter(kwargs.items()))
            self.dict.update(dict(
                dest = dest.strip('_'),
                const = const
            ))


        if help is not None:
            self.dict["help"] = help
 
        if islist:
            assert type
            self.dict["nargs"] = "*"

class StoreArgs:
    def __init__(self, **kwargs):
        self.args = [
            Arg(**{f"_{k}": v}) for k, v  in kwargs.items()
        ]


class AutoParser:
    def __init__(self,
                 args: List[Arg]):

        self.parser = argparse.ArgumentParser()

        for arg in args:
            if isinstance(arg, Arg):
                self.parser.add_argument(*arg.list, **arg.dict)
            else:
                for a in arg.args:
                    self.parser.add_argument(*a.list, **a.dict)

    def parse_args(self):
        return self.parser.parse_args()


if __name__ =="__main__":
    # args = AutoParser([
    #     Arg(haha = 1.),
    #     # Arg(name = "wtf", action = "store_false"),

    # ]).parse_args()

    # # print(args.eval)
    # print(args.haha, type(args.haha))
    # print(args.


    args = AutoParser([
        StoreArgs(
            micro_train_batch_size = 8,
            train_batch_size = 128,
            max_norm = 1.0,
            seed = 42,
            enable_bf16 = True,
            zpg = 1,
            overlap_comm = True,
            max_epochs = 2,
            learning_rate = 5e-6,
            lr_warmup_ratio = 0.03,
            l2 = 0,
            adam_betas = (0.9, 0.95),
            ring_attn_size = 4,
            ring_head_stride = 1,

        )

    ]).parse_args()

    print(args.train_batch_size)