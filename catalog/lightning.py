from lightning.vision.classification import ClassificationTrainer  # noqa E403
from lightning.vision.mmdetection import MMDetectionTrainer  # noqa E403

from ._get import _get


def get(name):
    return _get(globals(), name, "LightningModule")


def build(name, *args, **kwargs):
    return get(name)(*args, **kwargs)
