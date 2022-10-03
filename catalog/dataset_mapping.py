from data.mapping import *  # noqa

from ._get import _get


def get(name):
    return _get(globals(), name, "Dataset-mapping")


def build(name, args={}, *_args, **kwargs):
    return get(name)(*_args, **args, **kwargs)
