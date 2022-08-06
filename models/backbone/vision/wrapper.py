from typing import Optional

import timm
from torchvision import models
from utils.models import drop_layers_after


def TimmNetwork(
    name: str,
    reset_classifier: bool = True,
    drop_after: Optional[str] = None,
    args: Optional[dict] = {},
):
    """
    Load model(and pretrained-weights) implemented in `timm`.

    Model catalog can be found in: https://rwightman.github.io/pytorch-image-models/models

    Parameters
    ----------
    name: str
        Name of the corresponding timm model to use. timm.list_models() returns
        a complete list of available models in timm(https://timm.fast.ai).
    reset_classifier: bool
        Drops the classifier layer to use as a feature extractor.
    drop_after: str
        Remove all the layers including and after `drop_after` layer. For example, in the example below of `resnet50`,
        we want to set `drop_after` = "avgpool" to drop the final 2 layers and output feature map.
        ```
                ...
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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
        feature_extractor network that can be used in multiple subtasks by plugging in different downstream heads.
    """
    # detach final classification head(make it feature extractor)
    # find model with same id & create model
    model = timm.create_model(name, num_classes=0, **args)
    if reset_classifier:
        model.reset_classifier(0, "")
    # detach final classification head(make it feature extractor)
    if drop_after:
        model = drop_layers_after(model, drop_after)

    return model


def TorchvisionNetwork(
    name: str,
    drop_after: Optional[str] = None,
    args: Optional[dict] = {},
):
    """
    Load model(and pretrained-weights) implemented in `torchvision.models`. Although some of our custom
    architecture implementation is also sort of based on torchvision, we implement this method to support more
    models and access to pretrained checkpoints.

    Model catalog can be found in: https://pytorch.org/vision/stable/models.html#id1
    TODO: support `torchvision` models for other tasks. e.g. video.

    Parameters
    ----------
    name: str
        Name of model to use. We look for `torchvision.models.{model_id}`.
    drop_after: str
        Remove all the layers including and after `drop_after` layer. For example, in the example below of `resnet50`,
        we want to set `drop_after` = "avgpool" to drop the final 2 layers and output feature map.
        ```
                ...
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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
        feature_extractor network that can be used in multiple subtasks by plugging in different downstream heads.
    """
    # find model with same id & create model
    model = getattr(models, name)(**args)
    # detach final classification head(make it feature extractor)
    if drop_after:
        model = drop_layers_after(model, drop_after)
    return model
