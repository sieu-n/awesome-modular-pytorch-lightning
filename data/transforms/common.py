import torchvision.transforms.functional as TF
try:
    from utils.data_container import DataContainer
except ImportError:
    pass
from .base import _BaseTransform, _KeyTransform


class ToTensor(_KeyTransform):
    """
    keys: List[str]
        list of keys to convert.
    """

    def __init__(self, *args, **kwargs):
        super(ToTensor, self).__init__(*args, **kwargs)

    def transform(self, t):
        return TF.to_tensor(t)


class RemoveKeys(_BaseTransform):
    """
    keys: List[str]
        list of keys to remove.
    """
    def __init__(self, keys, *args, **kwargs):
        super(RemoveKeys, self).__init__(*keys, **kwargs)
        self.keys = keys

    def __call__(self, d):
        for k in self.keys:
            d.pop(k)
        return d


class RenameKeys(_BaseTransform):
    def __init__(self, mapper: dict, *args, **kwargs):
        super(RemoveKeys, self).__init__(*args, **kwargs)
        self.mapper = mapper

    def __call__(self, d):
        for k, v in self.mapper:
            d[v] = d.pop(k)


class CollectDataContainer(_KeyTransform):
    """
    keys: List[str]
        list of keys to wrap inside `DataContainer` object.
    """
    def __init__(
        self,
        cpu_only: bool = False,
        stack: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cpu_only = cpu_only
        self.stack = stack

    def __call__(self, D):
        return DataContainer(D, cpu_only=self.cpu_only, stack=self.stack)
