import torchvision
from ttach.base import DualTransform

from data.transforms.vision.util import str2interpolation


class Rotation(DualTransform):
    """Flip images horizontally (left->right)"""

    identity_param = False

    def __init__(self, angles, interpolation="bilinear"):
        super().__init__("angle", angles)
        self.interpolation = interpolation
        if type(interpolation) == str:
            self.interpolation = str2interpolation(interpolation)

    def apply_aug_image(self, image, angle, **kwargs):
        image = torchvision.transforms.functional.rotate(image, angle=angle)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        raise NotImplementedError()
        return mask

    def apply_deaug_label(self, label, apply=False, **kwargs):
        return label

    def apply_deaug_keypoints(self, keypoints, apply=False, **kwargs):
        raise NotImplementedError()
        return keypoints
