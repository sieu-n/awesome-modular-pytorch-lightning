# model catalog
from models.backbone.vision import * # noqa

# utils
from .._get import _get


def get(name):
    return _get(globals(), name, "Backbone")


def build(name, args={}, *_args, **kwargs):
    return get(name)(*_args, **args, **kwargs)
