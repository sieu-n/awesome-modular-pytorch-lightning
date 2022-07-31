import torch
from typing import Union, List


class ToTensor():
    """
    keys: List[str]
        list of keys to convert.
    """
    def __init__(self, keys: Union[None, List[str]] = None):
        self.keys = keys

    def __call__(self, d):
        if self.key is None:
            return torch.tensor(d)
        else:
            for k in self.keys:
                d[k] = torch.tensor(d[k], dtype=torch.float32)
            return d


class RemoveKeys():
    """
    keys: List[str]
        list of keys to remove.
    """
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, d):
        for k in self.keys:
            d.pop(k)
        return d
