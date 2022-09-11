from pytorch_lightning.callbacks import Callback
from torch.nn import Module
from typing import Callable, Union, List

from utils import rgetattr


class WeightInitialization(Callback):
    """
    Callback that re-initializes weights of certain layers of the nn. One of either `mode` or `init_fn`
    should be specified.

    Parameters
    ----------
    mode: str (optional)
        Key for type of initialization to apply. Currently includes ["const", "normal", "gaussian"]
        mode.
        When mode is not specified, `init_fn` should be specified.
    target: Union[str, List[str]] (optional)
        Specify submodule or list of submodules to apply initialization to. By default
        reinitializes all modules of the model. 
    init_fn: Callable (optional)
        Function that recieves nn.Module and reinitializes parameters.
        When init_fn is not specified, `mode` should be specified.
    """
    def __init__(self, mode: str = None, target: Union[str, List[str]] = None, init_fn: Callable = None, *args, **kwargs):
        mode_dict = {
            "const": self.const_init,
            "normal": self.gaussian_init,
            "gaussian": self.gaussian_init,
        }
        if isinstance(target, str):
            self.target = [target]
        else:
            self.target = target

        if mode is not None:
            mode = mode.lower()
            assert mode in self.mode_dict, f"Invalid mode '{mode}' given to `WeightInitialization` \
                callback was not defined in known modes: {mode_dict.keys()}"
            self.init_apply_fn = lambda module: mode_dict[mode](module, *args, **kwargs)
        elif init_fn is not None:
            self.init_apply_fn = init_fn
        else:
            raise ValueError("Either `mode` or `init_fn` must be specified for WeightInitialization callback.")

    def on_fit_start(self, trainer, pl_module):
        if self.target is None:
            pl_module.apply(self.init_apply_fn)
        else:
            module = rgetattr(pl_module, self.target)
            module.apply(self.init_apply_fn)

    def const_init(self, module: Module, weight: float, bias: float = None):
        if bias is None:
            bias = weight

        if hasattr(module, "weight"):
            module.weight.data.constant_(weight)
        if hasattr(module, "bias"):
            module.bias.data.constant_(weight)
        
    def gaussian_init(self, module, mean=0.0, var=1.0, bias=False):
        if hasattr(module, "weight"):
            module.weight.data.normal_(mean, var)
        if bias and hasattr(module, "bias"):
            module.bias.data.normal_(mean, var)
