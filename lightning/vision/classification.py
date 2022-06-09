import torch
from torchmetrics.functional import accuracy
from algorithms.augmentation.mixup import MixupCutmix

from lightning.base import _BaseLightningTrainer
from algorithms.rdrop import compute_kl_loss


class ClassificationTrainer(_BaseLightningTrainer):
    def __init__(self, model_cfg, training_cfg, *args, **kwargs):
        super().__init__(model_cfg, training_cfg, *args, **kwargs)
        # mixup for classification
        self.mixup_cutmix = None
        if "mixup_cutmix" in training_cfg:
            self.mixup_cutmix = MixupCutmix(**training_cfg["mixup_cutmix"])
        # rdrop: R-Drop: Regularized Dropout for Neural Networks, NeurIPS-2021
        if "rdrop" in training_cfg:
            self.rdrop = True
            # assert "kl_loss" in training_cfg, "R-Drop requires `kl_loss` to be defined."
            self.rdrop_alpha = training_cfg["rdrop"]["alpha"]
        # assert whether all modules are initialized
        for name in ["loss_fn", "backbone", "classifier"]:
            assert hasattr(self, name), f"{name} is not initialized! Check the config file."

    def forward(self, x):
        feature = self.backbone(x)
        return self.classifier(feature)

    def training_step(self, batch, batch_idx):
        assert "images" in batch
        assert "labels" in batch

        x, y = batch["images"], batch["labels"]
        if self.mixup_cutmix:
            x, y = self.mixup_cutmix(x, y)
        logits = self(x)

        loss = self.loss_fn(logits, y)

        # r-drop
        if hasattr(self, "rdrop") and self.rdrop is True:
            logits2 = self(x)
            loss2 = self.loss_fn(logits2, y)

            ce_loss = (loss + loss2) * 0.5
            kl_loss = compute_kl_loss(logits, logits2)

            self.log("step/ce_loss", ce_loss)
            self.log("step/kl_loss", kl_loss)
            loss = ce_loss + self.rdrop_alpha * kl_loss

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
