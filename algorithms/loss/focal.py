import torch
from torch import nn


class CohenKappaWeight(nn.Module):
    def __init__(self, num_classes, reduction="mean", label_smoothing=0.0):
        super(CohenKappaWeight, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        assert len(target.shape) < 3
        if len(target.shape) == 2:  # one-hot bs, nc
            target = torch.argmax(target, dim=1)

        cohen_kappa_weights = self.get_cohen_kappa_weights(target)
        loss = cohen_kappa_weights * torch.clamp(logits - self.label_smoothing, min=0.0)
        loss = loss.mean(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError()

    def get_cohen_kappa_weights(self, y):
        # only works for single label.
        def f(x):
            return [(i - x) ** 2 for i in range(self.num_classes)]

        x = map(f, y)
        return torch.tensor(list(x))
