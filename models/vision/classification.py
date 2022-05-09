import models.heads as TorchHeads
import torch.nn as nn
from models.build import build_backbone


class ClassificationModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.backbone = build_backbone(model_cfg["backbone"])
        # TODO: adaptively attach every element in model_cfg["heads"]
        classifier_cfg = model_cfg["heads"]["classifier"]
        self.classifier = getattr(TorchHeads, classifier_cfg["ID"])(
            in_features=model_cfg["backbone"]["out_features"],
            **classifier_cfg["cfg"],
        )

    def forward(self, x):
        feature = self.backbone(x)
        return self.classifier(feature)
