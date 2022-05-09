import timm

import models.vision.backbone as CustomModels
import torchvision.models as TorchvisionModels

from utils.models import drop_layers_after


def build_backbone(backbone_cfg):
    if backbone_cfg["TYPE"] == "torchvision":
        backbone = torchvision_feature_extractor(
            backbone_cfg["ID"],
            drop_after=backbone_cfg.get("drop_after", None),
            kwargs=backbone_cfg.get("cfg", {}),
        )
    elif backbone_cfg["TYPE"] == "timm":
        backbone = timm_feature_extractor(
            backbone_cfg["ID"],
            drop_after=backbone_cfg["drop_after"],
            **backbone_cfg["cfg"],
        )
    elif backbone_cfg["TYPE"] == "custom":
        kwargs = backbone_cfg.get("cfg", {})
        return getattr(CustomModels, str(backbone_cfg["ID"]))(**kwargs)
    else:
        raise ValueError(f"Invalid `model.backbone.TYPE`: `{backbone_cfg['TYPE']}")

    return backbone


def timm_feature_extractor(model_id, kwargs={}):
    """
    Load model(and pretrained-weights) implemented in `timm`.

    Model catalog can be found in: https://rwightman.github.io/pytorch-image-models/models

    Parameters
    ----------
    model_id: str
        Exact name of model to use. We look for `torchvision.models.{model_id}`.
    kwargs: dict
        kwargs for building model.
    Returns
    -------
    nn.Module
        feature_extractor network that can be used in multiple subtasks by plugging in different downstream heads.
    """
    # detach final classification head(make it feature extractor)
    kwargs["num_classes"] = 0
    kwargs["global_pool"] = ""
    # find model with same id & create model
    model = timm.create_model(model_id, **kwargs)
    return model


def torchvision_feature_extractor(model_id, drop_after=None, kwargs={}):
    """
    Load model(and pretrained-weights) implemented in `torchvision.models`. Although some of our custom
    architecture implementation is also sort of based on torchvision, we implement this method to support more
    models and access to pretrained checkpoints.

    Model catalog can be found in: https://pytorch.org/vision/stable/models.html#id1
    TODO: support `torchvision` models for other tasks. e.g. video.

    Parameters
    ----------
    model_id: str
        Exact name of model to use. We look for `torchvision.models.{model_id}`.
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
    kwargs: dict
        kwargs for building model.
    Returns
    -------
    nn.Module
        feature_extractor network that can be used in multiple subtasks by plugging in different downstream heads.
    """
    # find model with same id & create model
    model = getattr(TorchvisionModels, str(model_id))(**kwargs)
    # detach final classification head(make it feature extractor)
    if drop_after:
        drop_layers_after(model, drop_after)
    return model
