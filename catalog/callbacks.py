# callback catalog
from pytorch_lightning import callbacks as _PytorchLightningCallbacks

# utils
from ._get import _get_from_sources


def get(name, file=None):
    sources = {
        "lightning": _PytorchLightningCallbacks,
    }
    return _get_from_sources(
        sources=sources, scope=globals(), name=name, file=file, object_type="Lightning-callbacks"
    )


def build(name, file=None, *args, **kwargs):
    return get(name, file)(*args, **kwargs)
