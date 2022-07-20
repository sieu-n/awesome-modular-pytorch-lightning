import random

from torch.utils.data import Dataset


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
            print(f"`const_cfg` is not specified while initializing `{self}`. This might lead to unexpected behaviour.")
        else:
            self.const_cfg = const_cfg

    def __call__(self, x, y):
        raise NotImplementedError


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
        for t in self.transforms:
            d = t(d)
        return d


class RandomOrder:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, d):
        n_transforms = len(self.transforms)
        apply_order = random.shuffle(range(n_transforms))
        for idx in range(n_transforms):
            t = self.transforms[apply_order[idx]]
            d = t(d)
        return d
