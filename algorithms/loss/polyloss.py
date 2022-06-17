# Implement family of loss functions proposed in the paper:
# POLYLOSS: A Polynomial Expansion Perspective of Classification Loss Functions, ICLR 2022
import torch
from torch import nn


class PolyLoss(nn.CrossEntropyLoss):
    def __init__(self, eps, n=1, *args, **kwargs):
        """
        Implement general PolyLoss from the paper:
        POLYLOSS: A Polynomial Expansion Perspective of Classification Loss Functions, ICLR 2022

        Parameters
        ----------
        eps: float, list[float]
            Weights for each term of the taylor-expansion of the loss function.
        n: int, optional, default=1
            Degree of polynomial to add.
        *args, **kwargs: optional
            Arguments for the base nn.CrossEntropyLoss loss function.
        """
        super().__init__(*args, **kwargs)
        self.n = n
        if type(eps) == float:
            assert n == 1, "If n!=1 provide a list of epsilon values."
            eps = [eps]
        assert len(eps) == n, f"Expected `eps` to have length n: {n}, but got {len(eps)}."
        self.eps = eps

    def poly_loss(self, x, target, degree=1):
        # label smoothing
        num_classes = x.shape[1]
        smooth_labels = target * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        pt = (1 - torch.mean(smooth_labels * nn.functional.softmax(x), dim=1)).pow(degree)
        if self.reduction == "none":
            return pt
        elif self.reduction == "sum":
            return torch.sum(pt)
        elif self.reduction == "mean":
            return torch.mean(pt)
        else:
            raise ValueError(f"Invalid reduction value: {self.reduction}")

    def forward(self, x, target):
        CE = super().forward(x, target)
        pt = 0
        for deg in range(1, self.n + 1):
            pt += self.poly_loss(x, target, degree=deg) * self.eps[deg - 1]
        return CE + pt

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={self.n}, eps={self.eps}, reduction='{self.reduction}')"
