import torchmetrics

from ._get import _get_from_sources


def get(name, file=None):
    sources = {
        "torchmetrics": torchmetrics,
    }
    return _get_from_sources(
        sources=sources, scope=globals(), name=name, file=file, object_type="Metric"
    )
