import torchvision.transforms as TT
from .image import _ImageTransform
from torchvision.transforms.functional import InterpolationMode


class TorchvisionTransforms(_ImageTransform):
    def __init__(self, name, args={}, *_args, **kwargs):
        super().__init__(*_args, **kwargs)
        self.transform_f = getattr(TT, name)(**args)
        print(f"Found name `{name} from `torchvision.transforms`.")

    def transform(self, d):
        return self.transform_f(d)


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
