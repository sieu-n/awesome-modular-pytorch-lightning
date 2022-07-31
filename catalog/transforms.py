from data.transforms.common import RemoveKeys, ToTensor, CollectDataContainer  # noqa
from data.transforms.vision import *  # noqa

from ._get import _get


def get(name):
    return _get(globals(), name, "Transforms")


def build(name, args={}, *_args, **kwargs):
    return get(name)(*_args, **args, **kwargs)
