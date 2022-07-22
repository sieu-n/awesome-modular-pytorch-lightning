from data.dataset.mapping.noise import *  # noqa E403
from data.dataset.mapping.utils import *  # noqa E403

from ._get import _get


def get(name):
    return _get(globals(), name, "Dataset-mapping")


def build(name, *args, **kwargs):
    return get(name)(*args, **kwargs)
