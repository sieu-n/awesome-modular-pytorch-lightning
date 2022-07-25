from torch.nn import Module
from typing import Union, List, Optional

from .. import rsetattr, rhasattr, rgetattr
import catalog


def SetModule(model: Module, key: str, module_cfg: dict, allow_replace: bool = True):
    """Build a new module based on `module_cfg` and replace `model.key` with
    the newly created module.
    Parameters
    ----------
    model: Module
    key: str
    """
    new_module = catalog.modules.build(
        name=module_cfg["name"],
        file=module_cfg.get("file", None),
        **module_cfg.get("args", {}),
    )

    if rhasattr(model, key):
        assert allow_replace, f"model already has attribute `{key}`: \
            {rgetattr(model, key)}. set `allow_replace` to replace"
        print(f"Replacing {rgetattr(model, key)} -> {new_module}")
    else:
        print(f"Creating {key} attribute of model to {new_module}")

    rsetattr(model, key, new_module)


def ReplaceModulesOfType(
    model: Module,
    types: Union[List[str], str],
    target_type: str,
    files: Union[List[Optional[str]], Optional[str]] = None,
    target_file: Union[str, None] = None,
):
    """Replace all modules of type `types` to `target_type`. For example, change
    all `ReLU` activation functions to `Swish` function.
    TODO
    """
    pass


def ResNetLowResHead(model: Module, num_channels: int = 64, pooling: bool = True):
    """Replace input head of ResNet model to `low_res` variant so the model
    recieve 32x32 images such as CIFAR dataset as input.
    """
    conv1_cfg = {
        "name": "Conv2d",
        "file": "torch.nn",
        "args": {
            "in_channels": 3,
            "out_channels": num_channels,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "bias": False
        }
    }
    maxpool_cfg = {
        "name": "Identity",
        "file": "torch.nn",
    }

    SetModule(model, "backbone.conv1", conv1_cfg)
    if pooling:
        SetModule(model, "backbone.maxpool", maxpool_cfg)
