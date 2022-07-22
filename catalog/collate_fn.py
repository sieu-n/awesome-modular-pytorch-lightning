# collate function catalog
from data.collate.augmentation import timm_collate_mixup  # noqa E403
from data.collate.wrapper import mmcv_datacontainer_collate  # noqa E403
from torch.utils.data._utils.collate import default_collate  # noqa E403

# utils
from ._get import _get
from functools import partial


def get(name):
    return _get(globals(), name, "Collate-function")


def build(name, *args, **kwargs):
    return partial(get(name), *args, **kwargs)
