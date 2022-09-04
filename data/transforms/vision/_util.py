from torchvision.transforms.functional import InterpolationMode
from .. import _KeyTransform


class _ImageTransform(_KeyTransform):
    def __init__(self, *args, **kwargs):
        self.key = "images"  # set default key
        super(_ImageTransform, self).__init__(*args, **kwargs)


def str2interpolation(s):
    assert s in [
        "nearest",
        "bilinear",
        "bicubic",
        "box",
        "hamming",
        "lancoz",
    ], f"Expected key to have one of values \
                in the list, but got {s}"
    conversion_d = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "box": InterpolationMode.BOX,
        "hamming": InterpolationMode.HAMMING,
        "lancoz": InterpolationMode.LANCZOS,
    }
    return conversion_d[s]


def interpolation2str(i):
    conversion_d = {
        InterpolationMode.NEAREST: "nearest",
        InterpolationMode.BILINEAR: "bilinear",
        InterpolationMode.BICUBIC: "bicubic",
        InterpolationMode.BOX: "box",
        InterpolationMode.HAMMING: "hamming",
        InterpolationMode.LANCZOS: "lancoz",
    }
    return conversion_d[i]
