from algorithms.metrics import (  # noqa
    MPJPE,
    MMDet2TorchMetricmAP,
    SubsetMetric,
    TorchMetric,
)

from ._get import _get


def get(name):
    return _get(globals(), name, object_type="Metric")


def build(name, args={}, *_args, **kwargs):
    return get(name)(*_args, **args, **kwargs)
