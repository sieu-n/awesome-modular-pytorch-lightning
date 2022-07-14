# model catalog
from models.vision.backbone.resnet import ResNet18, ResNet34, ResNet50, ResNet101  # noqa F401

# utils
from models.vision.backbone.timm import timm_feature_extractor
from models.vision.backbone.torchvision import torchvision_feature_extractor
from .._get import _get


def get(name):
    return _get(globals(), name, "Model")


def build_backbone(name, model_type="custom", drop_after=None, *args, **kwargs):
    if model_type == "torchvision":
        backbone = torchvision_feature_extractor(
            model_id=name, drop_after=drop_after, *args, **kwargs
        )
    elif model_type == "timm":
        backbone = timm_feature_extractor(model_id=name, *args, **kwargs)
    elif model_type == "custom":
        return get(str(name))(**kwargs)
    else:
        raise ValueError(f"Invalid `model.backbone.TYPE`: `{model_type}")

    return backbone
