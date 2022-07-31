# dataset catalog
from data.dataset.vision import TorchvisionDataset, MMDetectionDataset # noqa
# utils
from ._get import _get


def get(name):
    return _get(globals(), name, "Dataset")


def build(name, args={}, *_args, **kwargs):
    return get(name)(*_args, **args, **kwargs)
