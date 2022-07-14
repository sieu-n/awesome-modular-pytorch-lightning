from torch import nn
from . import heads, loss

from .._get import _get_from_sources


def get(name, file=None):
    sources = {
        "heads": heads,
        "loss": loss,
        "torch.nn": nn,
    }
    return _get_from_sources(
        sources=sources,
        scope=globals(),
        name=name,
        file=file,
        object_type="Metric"
    )
