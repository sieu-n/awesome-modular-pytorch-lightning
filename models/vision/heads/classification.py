
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, input_c, num_classes, reduction="flatten", dropout=None, return_logits=True):
        """
        Basic classification head for various tasks.
        
        Parameters
        ----------
        input_c: int
        num_classes: int
        reduction: nn.Module or str, default="flatten", optional
        dropout: float between (0.0, 1.0), default=None, optional
        return_logits: bool, default=True, optional
        """
        super(ClassificationHead, self).__init__()
        # build `fc` layer.
        self.fc = nn.Linear(input_c, num_classes)
        # build `reduction` layer.
        if isinstance(reduction, nn.Module):
            self.reduction = reduction
        elif type(reduction) == str:
            if reduction == "gap":
                self.reduction = nn.AdaptiveAvgPool2d((1, 1))
            elif reduction == "flatten":
                self.reduction = nn.Flatten()
            elif reduction == "none":
                self.reduction = nn.Identity()
            else:
                raise ValueError(f"Invalid value for `reduction`: {self.reduction}")
        # build dropout.
        if dropout:
            assert 0.0 < dropout < 1.0
            self.dropout = nn.Dropout(p=dropout)
        # build activation.
        if not return_logits:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        # order: reduction -> dropout(optional) -> fc -> activation(optional).
        x = self.reduction(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.fc(x)
        if hasattr(self, "activation"):
            x = self.activation(x)
        return x
