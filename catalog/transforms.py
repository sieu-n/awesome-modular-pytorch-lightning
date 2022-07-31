from data.transforms.vision import *  # noqa
from data.transforms.common import ( # noqa
    ToTensor,
    RemoveKeys,
)


from ._get import _get


def get(name):
    return _get(globals(), name, "Transforms")


def build(name, args={}, *_args, **kwargs):
    return get(name)(*_args, **args, **kwargs)
