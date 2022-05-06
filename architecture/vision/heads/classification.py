
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, num_classes, reduction="flatten", dropout=None, return_logits=True):
        super(ClassificationHead, self).__init__()
        assert reduction in ["gap", "flatten", "adaptivepool"]

    def forward(x):


