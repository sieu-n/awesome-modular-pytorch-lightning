from typing import List, Union

import torch
try:
    from utils.data_container import DataContainer
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
        if self.keys is None:
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
    def __init__(
        self,
        keys: List[str],
        cpu_only: List[bool] = None,
        stack: List[bool] = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.keys = keys
        if cpu_only is None:
            cpu_only = [False] * len(keys)
        else:
            assert len(keys) == len(cpu_only), f"cpu_only must have length {len(keys)} but found {len(cpu_only)}"
        self.cpu_only = cpu_only
        if stack is None:
            stack = [False] * len(keys)
        else:
            assert len(keys) == len(stack), f"stack must have length {len(keys)} but found {len(stack)}"
        self.stack = stack

    def __call__(self, d):
        for idx, k in enumerate(self.keys):
            d[k] = DataContainer(d[k], cpu_only=self.cpu_only[idx])
        return d
