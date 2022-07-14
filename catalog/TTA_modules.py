from algorithms.tta.wrapper import ClassificationTTAWrapper, TTAFramework  # noqa E403

from ._get import _get


def get(name):
    return _get(globals(), name, "TTA-modules")
