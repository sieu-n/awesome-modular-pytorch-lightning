import pytorch_lightning as pl
from algorithms.optimizers.lr_scheduler.warmup import GradualWarmupScheduler
from algorithms.optimizers.sam import SAM
from torch import optim
from torch.optim import lr_scheduler


class _LightningModule(pl.LightningModule):
    def configure_optimizers(self):
        """
        Build optimizer, learning rate scheduler.
        Support the `lr_warmup` and `sharpness-aware` keywords.
        Sharpness-aware minimization for efficiently improving generalization, ICLR 2021
        """
        # optimizer
        optimizer_name = self.training_cfg["optimizer"]
        optimizer_kwargs = self.training_cfg["optimizer_cfg"]
        if optimizer_name == "sgd":
            optimizer_builder = optim.SGD
        elif optimizer_name == "adam":
            optimizer_builder = optim.Adam
        elif optimizer_name == "adamw":
            optimizer_builder = optim.AdamW
        else:
            raise ValueError(f"Invalid value for optimizer: {optimizer_name}")
        # apply sharpness-aware minimization optimizer
        if "sharpness-aware" in self.training_cfg:
            sam_cfg = self.training_cfg["sharpness-aware"]
            optimizer = SAM(
                params=self.parameters(),
                base_optimizer=optimizer_builder,
                rho=sam_cfg["rho"] if "rho" in sam_cfg else 0.05,
                **optimizer_kwargs,
            )
        else:
            optimizer = optimizer_builder(self.parameters(), **optimizer_kwargs)

        config = {"optimizer": optimizer}
        # lr schedule
        if "lr_scheduler" in self.training_cfg:
            scheduler_config = self.training_cfg["lr_scheduler"]
            schedule_name = scheduler_config["name"]
            schedule_kwargs = scheduler_config["args"]
            if schedule_name == "const":
                schedule_builder = lr_scheduler.LambdaLR
                schedule_kwargs["lr_lambda"] = lambda epoch: 1
            elif schedule_name == "cosine":
                schedule_builder = lr_scheduler.CosineAnnealingLR
                assert "frequency" not in scheduler_config or "frequency" == "epoch"
            elif schedule_name == "exponential":
                schedule_builder = lr_scheduler.ExponentialLR
            elif schedule_name == "1cycle":
                schedule_builder = lr_scheduler.OneCycleLR
            elif schedule_name == "step":
                schedule_builder = lr_scheduler.StepLR
            elif schedule_name == "multi-step":
                schedule_builder = lr_scheduler.MultiStepLR
            else:
                raise ValueError(f"Invalid value for lr_scheduler: {schedule_name}")
            scheduler = schedule_builder(optimizer, **schedule_kwargs)
            # learning rate warmup
            if "lr_warmup" in self.training_cfg:
                warmup_cfg = self.training_cfg["lr_warmup"]
                scheduler = GradualWarmupScheduler(
                    optimizer=optimizer, after_scheduler=scheduler, **warmup_cfg
                )
            # build scheduler with keys that lightning identifies.
            scheduler_config = scheduler_config["cfg"]
            if self.automatic_optimization is False:
                for k in ["frequency", "monitor", "strict"]:
                    assert k not in scheduler_config, f"{k} is not yet implemented!!!"
                assert "frequency" not in scheduler_config or scheduler_config[
                    "frequency"
                ] in [
                    "step",
                    "epoch",
                ], f"`frequency` should be one of [`step`, `epoch`] but {scheduler_config['frequency']} was given."
            config["lr_scheduler"] = {**{"scheduler": scheduler}, **scheduler_config}
        return config
