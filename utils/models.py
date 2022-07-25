import os

import torch.nn as nn


def has_layer(model, key, print_error=False):
    """
    Check if a layer exists from a `nn.Module` object. The layer is specified using a specific string format, which
    heirachically defines some layer.
    Parameters
    ----------
    model: nn.Module
        Base model object.
    key: str
        specifies what layer to get. for example, "layer4.1.conv2".
    """
    key = key.split(".")
    block = model
    for k in key:
        if not hasattr(block, k):
            if print_error:
                print(f"{block} has no attribute {k}.")
            return False
        block = getattr(block, k)
    return True


def find_layer(model, key):
    """
    Find and return a layer from a `nn.Module` object. The layer is specified using a specific string format, which
    heirachically defines some layer.
    Parameters
    ----------
    model: nn.Module
        Base model object.
    key: str
        specifies what layer to get. for example, "layer4.1.conv2".
    """
    key = key.split(".")
    block = model
    for k in key:
        block = getattr(block, k)
    return block


def drop_layers_after(base_model, key):
    """
    get feature_extractor network that can be used in multiple subtasks by plugging in different downstream heads
    from classification network.

    Parameters
    ----------
    base_model: nn.Module
        Base model object.
    drop_after: str
        Remove all the layers including and after `drop_after` layer. For example, in the example below of `resnet50`,
        we want can set `drop_after` = "avgpool" to drop the final 2 layers and output feature map.
        To define layers inside block, such as the Conv2D layer shown in the example below, use the slash(/) symbol:
            - `drop_after` = "layer4.1.conv2"
        ```
            (layer4): Sequential(
              ...
              (1): BasicBlock(
                ...
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) x
                (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
            (fc): Linear(in_features=512, out_features=1000, bias=True)
        )
        ```
    Returns
    -------
    nn.Module
        Trimmed network.
    """
    print(f"Trimming layers of model after {key}.")
    if os.environ["DEBUG_MODE"] == "TRUE":
        print("[*] Base model:", base_model)
    key = key.split(".")

    def trim_block(block, depth):
        child_blocks = list(block.children())
        to_trim_idx = child_blocks.index(getattr(block, key[depth]))
        if depth == len(key) - 1:
            return nn.Sequential(*child_blocks[:to_trim_idx])

        new_block = child_blocks[:to_trim_idx] + [
            trim_block(child_blocks[to_trim_idx], depth + 1)
        ]
        print(new_block)
        return nn.Sequential(*new_block)

    base_model = trim_block(base_model, 0)

    if os.environ["DEBUG_MODE"] == "TRUE":
        print("[*] Trimmed model:", base_model)
    return base_model
