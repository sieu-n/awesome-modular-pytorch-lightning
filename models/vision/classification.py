import torch.nn as nn
from models.util import build_backbone


class ClassificationModel(nn.Module):
    def __init__(self, model_cfg):
        self.backbone = build_backbone(model_cfg["backbone"])
        # TODO: adaptively attach every element in model_cfg["heads"]
        classifier_cfg = model_cfg["heads"]["classifier"]
        self.classifier = getattr(model_cfg, classifier_cfg["ID"])(**classifier_cfg)

    def forward(self, x):
        feature = self.backbone(x)
        return self.classifier(feature)
