from torch import nn

from .._get import _get_from_sources
from . import backbone, heads, loss


def get(name, file=None):
    sources = {
        "backbone": backbone,
        "heads": heads,
        "loss": loss,
        "torch.nn": nn,
    }
    return _get_from_sources(
        sources=sources, scope=globals(), name=name, file=file, object_type="Module"
    )


def build(name, file=None, args={}, *_args, **kwargs):
    return get(name, file)(*_args, **args, **kwargs)
