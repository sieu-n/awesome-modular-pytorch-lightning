import torch
import torchvision.transforms.functional as TF
from utils.data_container import DataContainer
from .base import _BaseTransform, _KeyTransform


class ToTensor(_KeyTransform):
    """
    keys: List[str]
        list of keys to convert.
    """

    def __init__(self, dtype=None, *args, **kwargs):
        super(ToTensor, self).__init__(*args, **kwargs)
        self.dtype = dtype

    def transform(self, t):
        dtype_map = {
            None: None,
            "float32": torch.float32,
            "float64": torch.float64,
            "uint8": torch.uint8,
            "bool": torch.bool,
            "long": torch.long,
            "int": torch.int,
            "float": torch.float,
        }
        return torch.tensor(t, dtype=dtype_map[self.dtype])


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

    def transform(self, D):
        return DataContainer(D, cpu_only=self.cpu_only, stack=self.stack)
