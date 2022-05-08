import pytorch_lightning as pl
from torch import optim
from torch.optim import lr_scheduler


class _BaseLightningTrainer(pl.LightningModule):
    def __init__(self, cfg, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.model = model

    def training_epoch_end(self, outputs):
        total_loss = sum([x["loss"] for x in outputs])
        total_loss = total_loss / len(outputs)
        self.log("trn_loss", float(total_loss.cpu()))

    def validation_epoch_end(self, validation_step_outputs):
        # TODO: make it flexible for more output formats.
        total_acc, total_loss = map(sum, zip(*validation_step_outputs))
        total_acc = total_acc / len(validation_step_outputs)
        total_loss = total_loss / len(validation_step_outputs)
        self.log("val_performance", total_acc)
        self.log("val_loss", total_loss)

    def test_epoch_end(self, test_step_outputs):
        total_acc = sum([x for x in test_step_outputs])
        total_acc = total_acc / len(test_step_outputs)
        self.log("test_performance", total_acc)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def evaluate(self, batch, stage=None):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def configure_optimizers(self):
        # optimizer
        optimizer_name = self.cfg["training"]["optimizer"]
        optimizer_kwargs = self.cfg["training"]["optimizer_cfg"]
        if optimizer_name == "sgd":
            optimizer_builder = optim.SGD
        elif optimizer_name == "adam":
            optimizer_builder = optim.Adam
        else:
            raise ValueError(f"Invalid value for optimizer: {optimizer_name}")
        optimizer = optimizer_builder(
            self.parameters(), lr=self.cfg["training"]["lr"], **optimizer_kwargs
        )
        config = {"optimizer": optimizer}
        # lr schedule
        if "lr_scheduler" in self.cfg["training"]:
            schedule_name = self.cfg["training"]["lr_scheduler"]
            schedule_kwargs = self.cfg["training"]["lr_scheduler_cfg"]
            if schedule_name == "const":
                schedule_builder = lr_scheduler.LambdaLR
                schedule_kwargs["lr_lambda"] = lambda epoch: 1
            elif schedule_name == "cosine":
                schedule_builder = lr_scheduler.CosineAnnealingLR
                schedule_kwargs["T_max"] = self.cfg["training"]["epochs"]
            elif schedule_name == "exponential":
                schedule_builder = lr_scheduler.ExponentialLR
            elif schedule_name == "step":
                schedule_builder = lr_scheduler.StepLR
            elif schedule_name == "multi-step":
                schedule_builder = lr_scheduler.MultiStepLR
            else:
                raise ValueError(f"Invalid value for lr_scheduler: {schedule_name}")
            scheduler = schedule_builder(optimizer, **schedule_kwargs)
            config["lr_scheduler"] = scheduler
        return config

    def attach_hook(self, model, layer):
        raise NotImplementedError()
