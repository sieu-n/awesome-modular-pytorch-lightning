from copy import deepcopy
from typing import Callable, List, Optional, Union

import catalog
from torch.nn import Module

from .. import rgetattr, rhasattr, rsetattr


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
        assert (
            allow_replace
        ), f"model already has attribute `{key}`: \
            {rgetattr(model, key)}. set `allow_replace` to replace"
        print(f"Replacing {rgetattr(model, key)} -> {new_module}")
    else:
        print(f"Creating {key} attribute of model to {new_module}")

    rsetattr(model, key, new_module)


def ReplaceModulesOfType(
    model: Module,
    subject_modules: Union[List[str], str],
    target_module: dict,
    subject_files: Union[List[Optional[str]], Optional[str]] = None,
    allow_subclass: bool = True,
    module_lambda_func: Callable = lambda module_cfg, layer: module_cfg,
):
    """Implementing customized network architectures from other frameworks such as `timm` can be
    tedious. `ReplaceModulesOfType` hook simplify experiments by replacing all modules of
    type `subject_modules` to a new module. For example, it can be used to
        - change `ReLU` activation functions to `Swish` function.
        - change batch normalization to group normalization / layer normalization + Weight standardization
        - redifine arguments of layers(e.g. replacing Conv2D -> Conv2D(..., kernel_size=5, ))
    Reference:
        - https://discuss.pytorch.org/t/how-to-replace-a-layer-with-own-custom-variant/43586/12
    """
    count = 0

    if type(subject_modules) == str:
        subject_modules = [catalog.modules.get(subject_modules, subject_files)]
    elif subject_files is None:
        subject_modules = [catalog.modules.get(t) for t in subject_modules]
    else:
        assert len(subject_files) == len(subject_modules)
        subject_modules = [
            catalog.modules.get(t, subject_files[idx])
            for idx, t in enumerate(subject_modules)
        ]

    for name, layer in model.named_modules():
        is_same_type = False
        for t in subject_modules:
            # `isinstance` is True not only when the types are exactly same but also
            # when `l` is a subclass of `target_type`.
            if allow_subclass and isinstance(layer, t):
                is_same_type = True
                break
            # types must be exactly same when `allow_subclass` is False.
            if not allow_subclass and type(layer) == t:
                is_same_type = True
                break

        if is_same_type:
            count += 1
            _target_module = module_lambda_func(deepcopy(target_module), layer)
            rsetattr(
                model,
                name,
                catalog.modules.build(
                    name=_target_module["name"],
                    file=_target_module.get("file", None),
                    **_target_module.get("args", {}),
                ),
            )

    print(
        f"`ReplaceModulesOfType` found and replaced {count} layers of type {subject_modules}."
    )


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
            "bias": False,
        },
    }
    maxpool_cfg = {
        "name": "Identity",
        "file": "torch.nn",
    }

    SetModule(model, "backbone.conv1", conv1_cfg)
    if pooling:
        SetModule(model, "backbone.maxpool", maxpool_cfg)
