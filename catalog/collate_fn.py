# collate function catalog
from torch.utils.data._utils.collate import default_collate # noqa E403
from data.collate.wrapper import mmcv_parallel_collate # noqa E403
from data.collate.augmentation import timm_collate_mixup # noqa E403

# utils
from ._get import _get


def get(name):
    return _get(globals(), name, "Collate-function")


def build(name, *args, **kwargs):
    return get(name)(*args, **kwargs)
