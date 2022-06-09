import math
import random
import warnings

import torchvision.transforms.functional as TF
from data.transforms.base import _BaseTransform
from data.transforms.vision.util import str2interpolation

_RANDOM_INTERPOLATION = (str2interpolation("bilinear"), str2interpolation("bicubic"))


class TupleToClassificationData(_BaseTransform):
    """
    initial transforms.
    """

    def __call__(self, x, y):
        return {"images": x, "labels": y}


class ClassificationLabelEncoder(_BaseTransform):
    def __init__(self, **kwargs):
        """
        Recieves the naive `VOC2012` torchvision dataset, which contains the annotations from the `.xml` file.
        Processes and returns the class and bbox in the following format:

        x = PIL.Image
        y = {"boxes": torch.Tensor(4, num_obj), "labels": torch.Tensor(num_obj)}

        Each bbox coordinates are given in (x, y, w, h) format. The numbers are normalized to (0, 1) range by dividing
        them with the width and height of the image.
        """
        super().__init__(**kwargs)
        self.label2code = {
            name: idx for idx, name in enumerate(self.const_cfg["label_map"])
        }

    def __call__(self, d):
        d["labels"] = self.label2code[d["labels"]]
        return d


class RandomResizedCropAndInterpolation(_BaseTransform):
    """
    Common procedure for affine data augmentation in ImageNet implemented in `timm`.
    Reference: https://bit.ly/3swswNE
    Crop the given PIL Image to random size and aspect ratio with random interpolation.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation="bilinear",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == "random":
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = str2interpolation(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, d):
        d["images"] = self.transform(d["images"])
        return d

    def transform(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return TF.resized_crop(img, i, j, h, w, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = " ".join(
                [str2interpolation(x) for x in self.interpolation]
            )
        else:
            interpolate_str = str2interpolation(self.interpolation)
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str})"
        return format_string
