# callback catalog
from algorithms.gradient_noise import (  # noqa
    AdditiveGradientNoiseCallback,
    DecayingGradientNoiseCallback,
)
from utils.callbacks import *  # noqa

# utils
from ._get import _get


def get(name):
    return _get(globals(), name, object_type="Lightning-callbacks")


def build(name, args={}, *_args, **kwargs):
    return get(name)(*_args, **args, **kwargs)
