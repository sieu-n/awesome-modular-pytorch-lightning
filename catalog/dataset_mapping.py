from data.mapping.noise import *  # noqa E403
from data.mapping.utils import *  # noqa E403

from ._get import _get


def get(name):
    return _get(globals(), name, "Dataset-mapping")


def build(name, args={}, *_args, **kwargs):
    return get(name)(*_args, **args, **kwargs)
