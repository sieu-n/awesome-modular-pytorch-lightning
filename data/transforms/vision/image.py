import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from data.transforms.common import _BaseTransform
from data.transforms.vision.util import str2interpolation


class _ImageTransform(_BaseTransform):
    def __call__(self, d):
        d["image"] = self.transform(d["image"])
        return d


class Normalize(_ImageTransform):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), **kwargs):
        """
        Normalize input image with predefined mean/std.

        Parameters
        ----------
        mean: list, len=3
            mean values of (r, g, b) channels to use for normalizing.
        std: list, len=3
            stddev values of (r, g, b) channels to use for normalizing.
        """
        super().__init__(**kwargs)
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, d):
        d["image"] = self.transform(d["image"])
        return d

    def transform(self, image):
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        return image


class ColorJitter(_ImageTransform):
    def __init__(
        self,
        brightness=None,
        contrast=None,
        saturation=None,
        hue=None,
    ):
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def transform(self, image):
        image = self.color_jitter(image)
        return image


class UnNormalize(_ImageTransform):
    def __init__(self, mean, std, **kwargs):
        """
        Inverse normalization given predefined mean/std values.
        reference: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/2

        Parameters
        ----------
        mean: list, len=3
            mean values of (r, g, b) channels to use for normalizing.
        std: list, len=3
            stddev values of (r, g, b) channels to use for normalizing.
        """
        super().__init__(**kwargs)
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def transform(self, tensor):
        """
        Parameters
        ----------
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns
        -------
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class ToTensor(_ImageTransform):
    def transform(self, image):
        return TF.to_tensor(image)


class ToPIL(_ImageTransform):
    def transform(self, image):
        return TF.to_pil_image(image)


class Resize(_ImageTransform):
    def __init__(
        self, size, interpolation="bilinear", max_size=None, antialias=None, **kwargs
    ):
        """Resize image to `size`, `image` should be torch.Tensor of [c, W, H]."""
        super().__init__(**kwargs)
        self.size = size
        self.interpolation = interpolation
        if type(self.interpolation) == str:
            self.interpolation = str2interpolation(self.interpolation)
        self.max_size = max_size
        self.antialias = antialias

    def transform(self, image):
        return TF.resize(
            image, self.size, self.interpolation, self.max_size, self.antialias
        )
