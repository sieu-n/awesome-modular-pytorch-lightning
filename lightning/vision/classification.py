import torch
import torch.nn as nn
from torchmetrics.functional import accuracy
from timm.data import Mixup

from lightning.base import _BaseLightningTrainer


class ClassificationTrainer(_BaseLightningTrainer):
    def __init__(self, model_cfg, training_cfg, *args, **kwargs):
        super().__init__(model_cfg, training_cfg, *args, **kwargs)
        # define loss function.
        if "losses" in training_cfg and "label_smoothing" in training_cfg["losses"]:
            label_smoothing = training_cfg["losses"]["label_smoothing"]
        else:
            label_smoothing = 0.0
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # mixup for classification
        self.mixup = None
        if "mixup_cutmix" in training_cfg:
            self.mixup = Mixup(**training_cfg["mixup_cutmix"])

    def forward(self, x):
        feature = self.backbone(x)
        return self.classifier(feature)

    def training_step(self, batch, batch_idx):
        assert "images" in batch
        assert "labels" in batch

        x, y = batch["images"], batch["labels"]
        if self.mixup:
            x, y = self.mixup(x, y)
        pred = self(x)

        loss = self.loss_fn(pred, y)
        self.log("step/train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        assert "images" in batch
        assert "labels" in batch

        x, y = batch["images"], batch["labels"]
        pred = self(x)
        loss = self.loss_fn(pred, y)
        class_pred = torch.argmax(pred, dim=1)
        acc = accuracy(class_pred, y)
        return loss, acc

    def predict_step(self, batch, batch_idx):
        assert "images" in batch
        x = batch["images"]
        pred = self(x)
        # class_pred = torch.argmax(pred, dim=1)
        return pred
