import torch.nn as nn
from models.vision.backbone import build_backbone


class FasterRCNNModel(nn.Module):
    # todo
    def __init__(self, model_cfg):
        super().__init__()
        self.backbone = build_backbone(model_cfg["backbone"])
        # TODO: adaptively attach every element in model_cfg["heads"]

    def forward(self, x):
        feature = self.backbone(x)
        return self.classifier(feature)
