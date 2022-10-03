# deprecated.
# utils
from .._get import _get
from ..modules.backbone import *  # noqa


def get(name):
    return _get(globals(), name, "Backbone")


def build(name, args={}, *_args, **kwargs):
    return get(name)(*_args, **args, **kwargs)
