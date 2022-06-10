from algorithms.augmentation.mixup import MixupCutmix
from algorithms.knowledge_distillation import TeacherModelKD
from lightning.base import _BaseLightningTrainer
from algorithms.rdrop import compute_kl_loss


class ClassificationTrainer(_BaseLightningTrainer):
    def __init__(self, model_cfg, training_cfg, *args, **kwargs):
        super().__init__(model_cfg, training_cfg, *args, **kwargs)
        # mixup and cutmix for classification
        self.mixup_cutmix = None
        if "mixup_cutmix" in training_cfg:
            self.mixup_cutmix = MixupCutmix(**training_cfg["mixup_cutmix"])
        # knowledge distillation using the method of: Distilling the Knowledge in a Neural Network, 2015
        if "kd" in training_cfg:
            kd_cfg = training_cfg["kd"]
            if "teacher" in kd_cfg:
                self.kd_inference = True
                self.teacher_model = TeacherModelKD(kd_cfg["teacher"])
            else:
                # use another way to recieve teacher model predictions.
                self.kd_inference = False
        # rdrop: R-Drop: Regularized Dropout for Neural Networks, NeurIPS-2021
        if "rdrop" in training_cfg:
            self.rdrop = True
            self.rdrop_alpha = training_cfg["rdrop"]["alpha"]
        # assert whether all modules are initialized
        for name in ["loss_fn", "backbone", "classifier"]:
            assert hasattr(self, name), f"{name} is not initialized! Check the config file."

    def forward(self, x):
        feature = self.backbone(x)
        return self.classifier(feature)

    def _traininig_step(self, batch, batch_idx):
        assert "images" in batch
        assert "labels" in batch

        x, y = batch["images"], batch["labels"]
        if self.mixup_cutmix:
            x, y = self.mixup_cutmix(x, y)
        logits = self(x)

        loss = self.loss_fn(logits, y)

        # r-drop
        if hasattr(self, "rdrop") and self.rdrop is True:
            loss = self.rdrop_forward(x, y, logits, loss)
        self.log("step/train_loss", loss)
        return {
            "pred": logits, "y": y, "loss": loss
        }

    def rdrop_forward(self, x, y, logits1, loss1):
        logits2 = self(x)
        loss2 = self.loss_fn(logits2, y)

        ce_loss = (loss1 + loss2) * 0.5
        kl_loss = compute_kl_loss(logits1, logits2)

        self.log("step/ce_loss", ce_loss)
        self.log("step/kl_loss", kl_loss)
        return ce_loss + self.rdrop_alpha * kl_loss

    def evaluate(self, batch, stage=None):
        assert "images" in batch
        assert "labels" in batch

        x, y = batch["images"], batch["labels"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return {
            "pred": logits, "y": y, "loss": loss
        }

    def predict_step(self, batch, batch_idx):
        assert "images" in batch
        x = batch["images"]
        pred = self(x)
        # class_pred = torch.argmax(pred, dim=1)
        return pred
