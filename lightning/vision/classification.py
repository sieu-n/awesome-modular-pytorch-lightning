import torch
import torch.nn as nn
from lightning.common import _BaseLightningTrainer
from torchmetrics.functional import accuracy


class ClassificationTrainer(_BaseLightningTrainer):
    def __init__(self, model_cfg, training_cfg, *args, **kwargs):
        super().__init__(model_cfg, training_cfg, *args, **kwargs)
        # define loss function.
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        feature = self.backbone(x)
        return self.classifier(feature)

    def training_step(self, batch, batch_idx):
        assert "images" in batch
        assert "labels" in batch

        x, y = batch["images"], batch["labels"]
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
