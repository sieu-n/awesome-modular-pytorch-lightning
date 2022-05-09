import torch
import torch.nn as nn
from lightning.common import _BaseLightningTrainer
from torchmetrics.functional import accuracy


class ClassificationTrainer(_BaseLightningTrainer):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        # define loss function.
        self.loss_fn = nn.NLLLoss()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)

        loss = self.loss_fn(pred, y)
        self.log("step/train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        class_pred = torch.argmax(pred, dim=1)
        acc = accuracy(class_pred, y)
        return loss, acc
