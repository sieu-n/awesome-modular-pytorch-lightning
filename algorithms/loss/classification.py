import torch
from torch import nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=x.shape[1])
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
