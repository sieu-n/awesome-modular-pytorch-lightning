import torchvision.transforms as TT
from data.transforms.common import _BaseTransform
from torchvision.transforms.functional import InterpolationMode


class TorchTransforms(_BaseTransform):
    def __init__(self, NAME, **kwargs):
        self.transform_f = getattr(TT, NAME)(**kwargs)
        print(f"[*] Found name `{NAME} from `torchvision.transforms`.")

    def input_transform(self, image):
        return self.transform_f(image)


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
