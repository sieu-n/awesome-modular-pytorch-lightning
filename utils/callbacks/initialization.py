from functools import partial
from typing import Callable, List, Union

from pytorch_lightning.callbacks import Callback
from torch.nn import Module
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

    def __init__(
        self,
        mode: str = None,
        target: Union[str, List[str]] = None,
        init_fn: Callable = None,
        layer_whitelist: list = None,
        *args,
        **kwargs,
    ):
        mode_dict = {
            "const": self.const_init,
            "normal": self.gaussian_init,
            "gaussian": self.gaussian_init,
        }
        self.layer_whitelist = layer_whitelist

        if isinstance(target, str):
            self.target = [target]
        else:
            self.target = target

        if mode is not None:
            mode = mode.lower()
            assert (
                mode in mode_dict
            ), f"Invalid mode '{mode}' given to `WeightInitialization` \
                callback was not defined in known modes: {mode_dict.keys()}"
            init_apply_fn = lambda module: mode_dict[mode](module, *args, **kwargs)
        elif init_fn is not None:
            init_apply_fn = init_fn
        else:
            raise ValueError(
                "Either `mode` or `init_fn` must be specified for WeightInitialization callback."
            )

        self.init_apply_fn = partial(self.check_layer_apply, init_apply_fn)

    def check_layer_apply(self, init_apply_fn, module):
        if (
            self.layer_whitelist is not None
            and module.__class__.__name__ not in self.layer_whitelist
        ):
            return

        init_apply_fn(module)

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
            if hasattr(module.weight, "data"):
                module.weight.data.normal_(mean, var)
            else:
                print(f"{module.name} has weight but no weight data.")
        if bias and hasattr(module, "bias"):
            if hasattr(module.bias, "data"):
                module.bias.data.normal_(mean, var)
            else:
                print(f"{module.name} has bias but no bias data.")
