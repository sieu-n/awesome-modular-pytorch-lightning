import torchvision.transforms as TT
from data.transforms.base import _BaseTransform
from torchvision.transforms.functional import InterpolationMode


class TorchTransforms(_BaseTransform):
    def __init__(self, NAME, ARGS={}, **kwargs):
        super().__init__(**kwargs)
        self.transform_f = getattr(TT, NAME)(**ARGS)
        print(f"Found name `{NAME} from `torchvision.transforms`.")

    def __call__(self, d):
        d["images"] = self.transform_f(d["images"])
        return d


def str2interpolation(s):
    assert s in ["nearest", "bilinear", "bicubic", "box", "hamming", "lancoz"]
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
