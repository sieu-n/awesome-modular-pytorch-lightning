import torch
from torch import nn


class CohenKappeWeightedCrossEntropy(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', label_smoothing=0.0):
        super(CohenKappeWeightedCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)

    def get_cohen_kappa_weights(y):
        # only works for single label.
        num_classes = y.shape[-1]
        weight_matrix = 