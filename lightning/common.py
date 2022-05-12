import pytorch_lightning as pl
from torch import optim
from torch.optim import lr_scheduler


class _BaseLightningTrainer(pl.LightningModule):
    def __init__(self, training_cfg, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.training_cfg = training_cfg
        self.model = model

    def training_epoch_end(self, outputs):
        total_loss = sum([x["loss"] for x in outputs])
        total_loss = total_loss / len(outputs)
        self.log("epoch/trn_loss", float(total_loss.cpu()))

    def validation_epoch_end(self, validation_step_outputs):
        # TODO: make it flexible for more output formats.
        total_loss, total_performance = map(sum, zip(*validation_step_outputs))
        total_performance = total_performance / len(validation_step_outputs)
        total_loss = total_loss / len(validation_step_outputs)
        self.log("epoch/val_performance", total_performance)
        self.log("epoch/val_loss", total_loss)

    def test_epoch_end(self, test_step_outputs):
        total_loss, total_performance = map(sum, zip(*test_step_outputs))
        total_performance = total_performance / len(test_step_outputs)
        total_loss = total_loss / len(test_step_outputs)
        self.log("epoch/test_performance", total_performance)
        self.log("epoch/test_loss", total_loss)

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
        optimizer_name = self.training_cfg["optimizer"]
        optimizer_kwargs = self.training_cfg["optimizer_cfg"]
        if optimizer_name == "sgd":
            optimizer_builder = optim.SGD
        elif optimizer_name == "adam":
            optimizer_builder = optim.Adam
        else:
            raise ValueError(f"Invalid value for optimizer: {optimizer_name}")
        optimizer = optimizer_builder(
            self.parameters(), lr=self.training_cfg["lr"], **optimizer_kwargs
        )
        config = {"optimizer": optimizer}
        # lr schedule
        if "lr_scheduler" in self.training_cfg:
            schedule_name = self.training_cfg["lr_scheduler"]
            schedule_kwargs = self.training_cfg["lr_scheduler_cfg"]
            if schedule_name == "const":
                schedule_builder = lr_scheduler.LambdaLR
                schedule_kwargs["lr_lambda"] = lambda epoch: 1
            elif schedule_name == "cosine":
                schedule_builder = lr_scheduler.CosineAnnealingLR
                schedule_kwargs["T_max"] = self.training_cfg["epochs"]
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
