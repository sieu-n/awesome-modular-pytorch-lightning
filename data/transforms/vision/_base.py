import warnings
from ..utils import _KeyTransform


class _ImageTransform(_KeyTransform):
    def __init__(self, key=None, *args, **kwargs):
        if key is None or key == "images":
            key = "images"  # set default key
        else:
            warnings.warn(f"Image transform {self} is intended to be applied to key `images` but \
                got key {key} instead.")
        super(_ImageTransform, self).__init__(key=key, *args, **kwargs)
