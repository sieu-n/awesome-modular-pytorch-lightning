from data.transforms.vision.classification import *  # noqa
from data.transforms.vision.detection import *  # noqa
from data.transforms.vision.image import *  # noqa
from data.transforms.vision.util import *  # noqa

from ._get import _get


def get(name):
    return _get(globals(), name, "Transforms")


def build(name, args={}, *_args, **kwargs):
    return get(name)(*_args, **args, **kwargs)
