import torch
import torch.nn.functional as F
from algorithms.augmentation.mixup import MixupCutmix
from algorithms.rdrop import compute_kl_loss
from lightning.base import _BaseLightningTrainer
from torch import nn


class ClassificationTrainer(_BaseLightningTrainer):
    def init(self, model_cfg, training_cfg):
        # mixup and cutmix for classification
        self.mixup_cutmix = None
        if "mixup_cutmix" in training_cfg:
            self.mixup_cutmix = MixupCutmix(**training_cfg["mixup_cutmix"])
        # rdrop: R-Drop: Regularized Dropout for Neural Networks, NeurIPS-2021
        if "rdrop" in training_cfg:
            self.rdrop = True
            self.rdrop_alpha = training_cfg["rdrop"]["alpha"]
        # initialize sub-loss functions. These functions must recieve (logits, y).
        sub_losses = []
        sub_loss_fns = training_cfg.get("sub_losses", {})
        for loss_name, loss_cfg in sub_loss_fns.items():
            loss_module = self.build_module(
                module_type=loss_cfg["name"],
                file=loss_cfg.get("file", None),
                **loss_cfg.get("args", {}),
            )
            sub_losses.append((loss_module, loss_cfg["weight"], loss_name))
        self.sub_losses = nn.ModuleList(sub_losses)
        # assert whether all modules are initialized
        for name in ["loss_fn", "backbone", "classifier"]:
            assert hasattr(
                self, name
            ), f"{name} is not initialized! Check the config file."

    def forward(self, x):
        feature = self.backbone(x)
        logits = self.classifier(feature)
        pred_prob = torch.nn.functional.log_softmax(logits, dim=1)
        return {
            "feature": feature,
            "logits": logits,
            "prob": pred_prob,
        }

    def _training_step(self, batch, batch_idx=0):
        assert "images" in batch
        assert "labels" in batch

        x, y = batch["images"], batch["labels"]
        if self.mixup_cutmix:
            x, y = self.mixup_cutmix(x, y)

        pred = self(x)
        logits = pred["logits"]

        loss = self.classification_loss(logits, y)

        # r-drop
        if hasattr(self, "rdrop") and self.rdrop is True:
            # loss = self.rdrop_forward(x, y, logits, loss)
            feature = pred["feature"]
            loss = self.fast_rdrop_forward(feature, y, logits, loss)

        self.log("step/train_loss", loss)

        # for logging
        res = {
            "y": y if y.ndim == 1 else y.argmax(dim=1),
            "logits": logits,
            "prob": pred["prob"],
            "cls_loss": loss,
        }

        assert "loss" not in res
        res["loss"] = loss
        return loss, res

    def evaluate(self, batch, stage=None):
        assert "images" in batch
        assert "labels" in batch

        x, y = batch["images"], batch["labels"]
        pred = self(x)
        logits = pred["logits"]

        loss = self.classification_loss(logits, y)
        return {
            "logits": logits,
            "prob": pred["prob"],
            "y": y,
            "cls_loss": loss,
        }

    def _predict_step(self, batch, batch_idx=0):
        assert "images" in batch
        x = batch["images"]
        pred = self(x)
        return pred["logits"]

    def classification_loss(self, logits, y):
        loss = self.loss_fn(logits, y)
        self.log("step/main_loss", loss)
        # add multiple losses defined in `training.loss_fn`
        for sub_loss_fn, weight, name in self.sub_losses:
            val = sub_loss_fn(logits, y)
            self.log(f"step/{name}_loss", val)
            loss += val * weight
        return loss

    def rdrop_forward(self, x, y, logits1, loss1):
        logits2 = self(x)
        loss2 = self.classification_loss(logits2, y)

        ce_loss = (loss1 + loss2) * 0.5
        kl_loss = compute_kl_loss(logits1, logits2)

        self.log("step/rdrop_ce_loss", ce_loss)
        self.log("step/rdrop_kl_loss", kl_loss)
        return ce_loss + self.rdrop_alpha * kl_loss

    def fast_rdrop_forward(self, feature, y, logits1, loss1):
        """Unofficial faster version of R-Drop which shares features."""
        logits2 = self.classifier(feature)
        loss2 = self.classification_loss(logits2, y)

        ce_loss = (loss1 + loss2) * 0.5
        kl_loss = compute_kl_loss(logits1, logits2)

        self.log("step/rdrop_ce_loss", ce_loss)
        self.log("step/rdrop_kl_loss", kl_loss)
        return ce_loss + self.rdrop_alpha * kl_loss
