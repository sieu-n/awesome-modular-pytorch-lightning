import torch
import torch.nn as nn
from lightning.common import _BaseLightningTrainer
from torchmetrics.functional import accuracy
import models.heads as TorchHeads
from models.vision.backbone import build_backbone


class ClassificationTrainer(_BaseLightningTrainer):
    def __init__(self, model_cfg, training_cfg, *args, **kwargs):
        super().__init__(training_cfg, *args, **kwargs)
        # define model
        self.backbone = build_backbone(model_cfg["backbone"])
        # TODO: adaptively attach every element in model_cfg["heads"]
        classifier_cfg = model_cfg["heads"]["classifier"]
        self.classifier = getattr(TorchHeads, classifier_cfg["ID"])(
            in_features=model_cfg["backbone"]["out_features"],
            **classifier_cfg["cfg"],
        )

        # define loss function.
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        feature = self.backbone(x)
        return self.classifier(feature)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        loss = self.loss_fn(pred, y)
        self.log("step/train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        class_pred = torch.argmax(pred, dim=1)
        acc = accuracy(class_pred, y)
        return loss, acc
