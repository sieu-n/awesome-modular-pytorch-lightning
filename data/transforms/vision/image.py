import torchvision.transforms.functional as TF

from data.transforms.vision.utils import _BaseTransform


class Normalize(_BaseTransform):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        """Normalize input image with predefined mean/std."""
        self.mean = mean
        self.std = std

    def image_transform(self, image):
        return image.sub_(self.mean).div_(self.std)


class UnNormalize(object):
    def __init__(self, mean, std):
        """
        Inverse normalization given mean/std values.
        reference: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/2
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
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


class ToTensor(_BaseTransform):
    def image_transform(self, image):
        return TF.to_tensor(image)


class ToPIL(_BaseTransform):
    def image_transform(self, image):
        return TF.to_pil_image(image)


'''
class ResizeImage(_BaseTransform):
    def __init__(self, size, interpolation="bilinear", max_size=None, antialias=None):
        """Resize image to `size`, `image` should be torch.Tensor of [c, W, H]."""
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def forward(self, image):
        return TF.resize()
'''
