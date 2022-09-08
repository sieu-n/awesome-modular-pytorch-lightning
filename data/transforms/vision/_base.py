from ..utils import _KeyTransform


class _ImageTransform(_KeyTransform):
    def __init__(self, *args, **kwargs):
        self.key = "images"  # set default key
        super(_ImageTransform, self).__init__(*args, **kwargs)
