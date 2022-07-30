# list of criterions(loss functions) for knowledge distillation
from algorithms.distillation.criterion import ( # noqa
    LogitKLCriterion,
    LogitMSECriterion,
)

from ._get import _get


def get(name):
    return _get(globals(), name, object_type="Distillation")


def build(name, args={}, *_args, **kwargs):
    return get(name)(*_args, **args, **kwargs)
