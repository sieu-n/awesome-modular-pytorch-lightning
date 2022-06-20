# modification of original Merger from `ttach`:
# https://github.com/qubvel/ttach/blob/94e579e59a21cbdfbb4f5790502e648008ecf64e/ttach/base.py#L120
from ttach import functional as F
import torch
from torch import nn


class Merger(nn.Module):
    def __init__(
        self,
        type: str = "mean",
        n: int = 1,
        num_classes: int = None,
        weight_per_class: bool = False,
    ):

        if type not in ["mean", "gmean", "sum", "max", "min", "tsharpen"]:
            raise ValueError("Not correct merge type `{}`.".format(type))

        self.output = None
        self.type = type
        self.n = n

        self.weight_per_class = weight_per_class
        if weight_per_class:
            assert type(num_classes) == int and num_classes > 1
            self.weights = nn.Parameter(data=torch.ones(n, num_classes), requires_grad=True)
        else:
            self.weights = nn.Parameter(data=torch.ones(n), requires_grad=True)

    def update(self, x, i=0):
        if self.weight_per_class:
            weight = self.weights[i]
        else:
            weight = self.weights[i]

        if self.type == "tsharpen":
            x = x**0.5

        if self.output is None:
            self.output = x
        elif self.type in ["mean", "sum", "tsharpen"]:
            self.output = self.output + x * weight
        elif self.type == "gmean":
            self.output = self.output * x
        elif self.type == "max":
            self.output = F.max(self.output, x)
        elif self.type == "min":
            self.output = F.min(self.output, x)

    def reset(self):
        self.output = None

    def compute(self):
        if self.type in ["sum", "max", "min"]:
            result = self.output
        elif self.type in ["mean", "tsharpen"]:
            result = self.output / self.n
        elif self.type in ["gmean"]:
            result = self.output ** (1 / self.n)
        else:
            raise ValueError("Not correct merge type `{}`.".format(self.type))
        return result
