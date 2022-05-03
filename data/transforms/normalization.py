import torchvision.transforms.functional as TF

"""
To support all tasks and augmentation, we recieve `y` in `__call__` and apply appropriate transformations to `y`.
"""


class ImageNormalization():
    """Normalize input image with predefined mean/std."""
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, x, y):
        return TF.normalize(x, self.mean, self.std), y


class InverseNormalize(object):
    def __init__(self, mean, std):
        """
        Inverse the normalization, we might want to do this before visualization.
        Reference: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor, y=None):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        if y:
            return tensor, y
        else:
            return tensor
