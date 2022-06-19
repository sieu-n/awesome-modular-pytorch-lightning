import torchvision

from ttach.base import DualTransform


class Rotation(DualTransform):
    """Flip images horizontally (left->right)"""

    identity_param = False

    def __init__(self, angles):
        super().__init__("angle", angles)

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
