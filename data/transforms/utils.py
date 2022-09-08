import random
import traceback

from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode


class _BaseTransform:
    """
    To support all tasks and augmentation, we recieve `x`, `y` in `__call__` and apply appropriate transformations to
    both `x` and `y`. This is useful important for many tasks such as
        - object detection
        - segmentation
    If you wish to apply transforms to only one of image or label, you may override one of `image_transform` or
    `label_transform` instead of overriding `__call__`.
    """

    def __init__(self, const_cfg=None):
        if const_cfg is None:
            print(
                f"`const_cfg` was not specified while initializing `{self}`. This might lead to unexpected behaviour."
            )
        else:
            self.const_cfg = const_cfg

    def __call__(self, d):
        raise NotImplementedError


class _KeyTransform(_BaseTransform):
    """
    Apply transform to certain key. Implement transformation by overriding `transform`.
    """

    def __init__(self, key=None, *args, **kwargs):
        super(_KeyTransform, self).__init__(*args, **kwargs)
        if key is not None:
            self.key = key
        elif hasattr(self, "key"):  # default key is defined
            pass
        else:
            self.key = None

    def transform(self):
        raise NotImplementedError

    def __call__(self, d):
        """
        Parameters
        ----------
        d: dict
        """
        if self.key is None:
            d = self.transform(d)
        else:
            d[self.key] = self.transform(d[self.key])
        return d


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


class ApplyTransforms(Dataset):
    def __init__(self, base_dataset, initial_transform=None, transforms=None):
        """
        Build dataset that simply adds more data transformations to the original samples.
        Parameters
        ----------
        base_dataset: torch.utils.data.Dataset
            base dataset that is used to get source samples.
        """
        self.base_dataset = base_dataset
        self.initial_transform = initial_transform
        self.transforms = transforms

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        d = self.base_dataset[idx]
        if self.initial_transform is not None:
            d = self.initial_transform(*d)
        if self.transforms is not None:
            d = self.transforms(d)
        return d


class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, d):
        for idx, t in enumerate(self.transforms):
            try:
                d = t(d)
            except:  # noqa
                traceback.print_exc()
                print(
                    f"Error occured during transformation {idx} / {len(self.transforms)}: {t}"
                )
                exit()
        return d


class RandomOrder:
    """
    Compose transforms but in a random order
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, d):
        n_transforms = len(self.transforms)
        apply_order = random.shuffle(range(n_transforms))
        for idx in range(n_transforms):
            t = self.transforms[apply_order[idx]]
            d = t(d)
        return d


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
