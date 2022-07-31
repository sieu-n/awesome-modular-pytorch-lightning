from typing import List, Union

import torch
try:
    from mmcv.parallel import DataContainer
except ImportError:
    pass
from .base import _BaseTransform


class ToTensor(_BaseTransform):
    """
    keys: List[str]
        list of keys to convert.
    """

    def __init__(self, keys: Union[None, List[str]] = None, *args, **kwargs):
        super(ToTensor, self).__init__(*args, **kwargs)
        self.keys = keys

    def __call__(self, d):
        if self.key is None:
            return torch.tensor(d)
        else:
            for k in self.keys:
                d[k] = torch.tensor(d[k], dtype=torch.float32)
            return d


class RemoveKeys(_BaseTransform):
    """
    keys: List[str]
        list of keys to remove.
    """

    def __init__(self, keys: List[str], *args, **kwargs):
        super(RemoveKeys, self).__init__(*args, **kwargs)
        self.keys = keys

    def __call__(self, d):
        for k in self.keys:
            d.pop(k)
        return d


class CollectDataContainer(_BaseTransform):
    """
    keys: List[str]
        list of keys to wrap inside `DataContainer` object.
    """
    def __init__(self, keys: List[str], *args, **kwargs):
        super(RemoveKeys, self).__init__(*args, **kwargs)
        self.keys = keys

    def __call__(self, d):
        for k in self.keys:
            d[k] = DataContainer(d[k])
        return d
