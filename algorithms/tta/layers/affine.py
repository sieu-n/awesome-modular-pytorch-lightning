import torchvision
from data.transforms.utils import str2interpolation
from ttach.base import DualTransform


class Rotation(DualTransform):
    """Flip images horizontally (left->right)"""

    identity_param = False

    def __init__(self, angles, interpolation="bilinear"):
        super().__init__("angle", angles)
        self.interpolation = interpolation
        if type(interpolation) == str:
            self.interpolation = str2interpolation(interpolation)

    def apply_aug_image(self, image, angle, **kwargs):
        image = torchvision.transforms.functional.rotate(
            image, angle=angle, interpolation=self.interpolation
        )
        return image

    def apply_deaug_mask(self, mask, angle, **kwargs):
        raise NotImplementedError()
        return mask

    def apply_deaug_label(self, label, angle, **kwargs):
        return label

    def apply_deaug_keypoints(self, keypoints, angle, **kwargs):
        raise NotImplementedError()
        return keypoints


class CenterZoom(DualTransform):
    """Flip images horizontally (left->right)"""

    identity_param = False

    def __init__(self, ratios, interpolation="bilinear"):
        super().__init__("ratio", ratios)
        self.interpolation = interpolation
        if type(interpolation) == str:
            self.interpolation = str2interpolation(interpolation)

    def apply_aug_image(self, image, ratio, **kwargs):
        """
        Zoom into image. This is implemented using the following procedure:
            1. centercrop
            2. resize to original size
        """
        input_size = (image.shape[-2], image.shape[-1])
        output_size = (int(image.shape[-2] / ratio), int(image.shape[-1] / ratio))
        image = torchvision.transforms.functional.center_crop(
            image,
            output_size=output_size,
        )
        image = torchvision.transforms.functional.resize(
            image, size=input_size, interpolation=self.interpolation
        )
        return image

    def apply_deaug_mask(self, mask, ratio, **kwargs):
        raise NotImplementedError()
        return mask

    def apply_deaug_label(self, label, ratio, **kwargs):
        return label

    def apply_deaug_keypoints(self, keypoints, ratio, **kwargs):
        raise NotImplementedError()
        return keypoints
