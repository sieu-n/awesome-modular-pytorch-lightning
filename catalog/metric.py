from algorithms.metrics import MPJPE, SubsetMetric, TorchMetric, MMDet2TorchMetricmAP  # noqa

from ._get import _get


def get(name):
    return _get(globals(), name, object_type="Metric")


def build(name, args={}, *_args, **kwargs):
    return get(name)(*_args, **args, **kwargs)
