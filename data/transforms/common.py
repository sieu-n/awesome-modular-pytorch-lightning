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

    def __call__(self, x, y):
        input_kwargs, label_kwargs = self.set_transform(x, y)
        return self.input_transform(x, **input_kwargs), self.label_transform(y, **label_kwargs)

    def set_transform(self, image, label):
        return {}, {}

    def input_transform(self, image):
        return image

    def label_transform(self, label):
        return label


class ApplyDataTransformations(Dataset):
    def __init__(self, base_dataset, transforms):
        """
        Build dataset that simply adds more data transformations to the original samples.
        Parameters
        ----------
        base_dataset: torch.utils.data.Dataset
            base dataset that is used to get source samples.
        """
        self.base_dataset = base_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        return self.transforms(x, y)


class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):
        for idx, t in enumerate(self.transforms):
            x, y = t(x, y)
        return x, y


class RandomOrder:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):
        for idx, t in enumerate(self.transforms):
            x, y = t(x, y)
        return x, y
