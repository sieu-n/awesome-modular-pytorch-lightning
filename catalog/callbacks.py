# callback catalog
from utils.callbacks import FreezeModule, LightningCallback  # noqa

# utils
from ._get import _get


def get(name):
    return _get(globals(), name, object_type="Lightning-callbacks")


def build(name, file=None, *args, **kwargs):
    return get(name, file)(*args, **kwargs)
