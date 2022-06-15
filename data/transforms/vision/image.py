import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from data.transforms.base import _BaseTransform
from data.transforms.vision.util import str2interpolation
from PIL import Image


class _ImageTransform(_BaseTransform):
    def __call__(self, d):
        d["images"] = self.transform(d["images"])
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


class CutOut(_ImageTransform):
    def __init__(
        self,
        mask_size=0.3,
        num_masks=1,
        p=0.5,
        cutout_inside=False,
        mask_color=0,
        **kwargs
    ):
        """
        https://github.com/hysts/pytorch_cutout/blob/ca4711283c7bc797774d486c6c41e06714350ded/dataloader.py#L36
        Improved regularization of convolutional neural networks with cutout.
        """
        super().__init__(**kwargs)
        assert p >= 0.0 and p <= 1.0
        assert mask_size >= 0.0 and mask_size <= 1.0
        self.p = p
        self.num_masks = num_masks
        self.cutout_inside = cutout_inside
        self.mask_color = mask_color

        self.mask_size_relative = mask_size

    def transform(self, image):
        if np.random.random() > self.p:
            return image

        image = np.asarray(image).copy()
        h, w = image.shape[:2]
        mask_size_x, mask_size_y = int(self.mask_size_relative * w), int(
            self.mask_size_relative * h
        )
        mask_size_x_half, mask_size_y_half = mask_size_x // 2, mask_size_y // 2
        offset_x, offset_y = (
            1 if mask_size_x % 2 == 0 else 0,
            1 if mask_size_y % 2 == 0 else 0,
        )

        for _ in range(self.num_masks):
            if self.cutout_inside:
                # cut region is fully inside image
                cxmin, cxmax = mask_size_x_half, w + offset_x - mask_size_x_half
                cymin, cymax = mask_size_y_half, h + offset_y - mask_size_y_half
            else:
                cxmin, cxmax = 0, w + offset_x
                cymin, cymax = 0, h + offset_y

            cx = np.random.randint(cxmin, cxmax)
            cy = np.random.randint(cymin, cymax)
            xmin = cx - mask_size_x_half
            ymin = cy - mask_size_y_half
            xmax = xmin + mask_size_x
            ymax = ymin + mask_size_y

            # clip within image boundary
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            # apply mask
            image[ymin:ymax, xmin:xmax, :] = self.mask_color
        return Image.fromarray(image)


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
