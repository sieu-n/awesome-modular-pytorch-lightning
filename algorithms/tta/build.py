from . import layers as TTA_layers
import ttach


def build_transforms(transforms):
    t = []
    for transform in transforms:
        t.append(_build_transform(
            name=transform["name"],
            **transform.get("args", {}),
        ))
    return ttach.Compose(t)


def _build_transform(name, **kwargs):
    if type(name) == str:
        return getattr(TTA_layers, name)(**kwargs)
    else:
        return name
