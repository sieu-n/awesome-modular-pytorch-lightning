from copy import deepcopy
from typing import List

import torch
from utils.data_container import DataContainer

from .base import _BaseTransform, _KeyTransform


class ToTensor(_KeyTransform):
    """
    Convert content of dictionary to `torch.Tensor` object.

    Parameters
    ----------
    keys: List[str]
        list of keys to convert.
    dtype: object, optional
        Target dtype of tensor to convert.
    channel_axis: List[int], optional
        When specified, the order of channels will be changed so these channels
        appear first.
    """

    def __init__(
        self, dtype: object = None, channel_axis: List[int] = None, *args, **kwargs
    ):
        super(ToTensor, self).__init__(*args, **kwargs)
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
        if dtype in dtype_map:
            self.dtype = dtype_map[dtype]
        else:
            self.dtype = dtype
        self.channel_axis = channel_axis

    def transform(self, t):
        t = torch.tensor(t, dtype=self.dtype)
        if self.channel_axis is not None:
            dim_order = self.channel_axis + list(
                set(range(t.ndim)) - set(self.channel_axis)
            )
            t = torch.permute(t, dim_order)
        return t


class RemoveKeys(_BaseTransform):
    """
    keys: List[str]
        list of keys to remove.
    """

    def __init__(self, keys, *args, **kwargs):
        super(RemoveKeys, self).__init__(*args, **kwargs)
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


class CopyKey(_BaseTransform):
    def __init__(self, a: dict, b: dict, *args, **kwargs):
        super(CopyKey, self).__init__(*args, **kwargs)
        self.a, self.b = a, b

    def __call__(self, d):
        d[self.b] = deepcopy(d[self.a])


class CollectDataContainer(_KeyTransform):
    """
    keys: List[str]
        list of keys to wrap inside `DataContainer` object.
    """

    def __init__(self, cpu_only: bool = False, stack: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpu_only = cpu_only
        self.stack = stack

    def transform(self, D):
        return DataContainer(D, cpu_only=self.cpu_only, stack=self.stack)
