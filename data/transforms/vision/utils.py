import torchvision.transforms as TT

from data.transforms.utils import _BaseTransform


class TorchTransforms(_BaseTransform):
    def __init__(self, NAME, **kwargs):
        self.transform_f = TT.__dict__[NAME](**kwargs)
        print(f"[*] Found name `{NAME} from `torchvision.transforms`.")

    def input_transform(self, image):
        return self.transform_f(image)
