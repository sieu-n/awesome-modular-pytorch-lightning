import catalog
import ttach


def build_transforms(transforms):
    t = []
    for transform in transforms:
        t.append(
            _build_transform(
                name=transform["name"],
                **transform.get("args", {}),
            )
        )
    return ttach.Compose(t)


def _build_transform(name, **kwargs):
    if type(name) == str:
        return catalog.TTA_layers.get(name)(**kwargs)
    else:
        return name
