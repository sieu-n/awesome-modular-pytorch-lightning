from collections import OrderedDict
import pytorch_lightning as pl
from torch import optim
from torch import nn
from torch.optim import lr_scheduler
from torch_ema import ExponentialMovingAverage

from algorithms.optimizers.lr_scheduler.warmup import GradualWarmupScheduler
from models import heads as HeadPool
from algorithms import loss as LossPool
from models.vision.backbone.timm import timm_feature_extractor
from models.vision.backbone.torchvision import torchvision_feature_extractor
from utils.models import get_layer
from utils.pretrained import load_model_weights


class _BaseLightningTrainer(pl.LightningModule):
    def __init__(self, model_cfg, training_cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # save training_cfg for defining optimizers when `configure_optimizers` is called.
        self.training_cfg = training_cfg
        # build backbone
        if "backbone" in model_cfg:
            backbone_cfg = model_cfg["backbone"]
            self.backbone = self.build_backbone(
                name=backbone_cfg["ID"],
                model_type=backbone_cfg["TYPE"],
                drop_after=backbone_cfg.get("drop_after", None),
                **backbone_cfg.get("cfg", {}),
            )
            # load backbone weights from url / filepath
            if "weights" in backbone_cfg:
                self.backbone = load_model_weights(
                    model=self.backbone, **backbone_cfg["weights"]
                )
        # build modules
        print("[*] Building modules attached to the backbone model...")
        modules = model_cfg.get("modules", {})
        for module_name, module_cfg in modules.items():
            head_module = self.build_module(
                module_type=module_cfg["name"],
                file=module_cfg.get("file", None),
                **module_cfg.get("args", {}),
            )
            setattr(self, module_name, head_module)
        # setup hooks
        self._hook_cache = {}
        hooks = model_cfg.get("hooks", {})
        for hook_name, hook_cfg in hooks.items():
            self.setup_hook(
                network=getattr(self, hook_cfg["model"]),
                key=hook_name,
                layer_name=hook_cfg["layer_name"],
                **hook_cfg.get("cfg", {}),
            )
        # build EMA
        self.EMA = "ema" in training_cfg
        if "ema" in training_cfg:
            ema_cfg = training_cfg["ema"]
            self.ema_manager = ExponentialMovingAverage(
                self.parameters(), decay=ema_cfg["decay"]
            )

    def build_module(self, module_type, file=None, *args, **kwargs):
        # build and return any nn.Module that is defined under `module_locations`.
        module_pool = OrderedDict({
            "heads": HeadPool,
            "loss": LossPool,
            "torch.nn": nn,
        })
        # if library is specified
        if file:
            if type(module_type) == str:
                module_type = getattr(module_pool[file], module_type)
            else:
                raise ValueError("If `to_look_at` is specified, provide the name of the module as a string.")
        elif type(module_type) == str:
            is_found = False
            for location in module_pool.values():
                if hasattr(module_type, location):
                    print(f"'{module_type}' was found in `{location}.")
                    module_type = getattr(location, module_type)
                    is_found = True
                    break
            if not is_found:
                raise ValueError(f"{module_type} was not found in the pool of modules: {list(module_pool.values())}")
        head = module_type(*args, **kwargs)
        return head

    def build_backbone(
        self, name, model_type="custom", drop_after=None, *args, **kwargs
    ):
        if model_type == "torchvision":
            backbone = torchvision_feature_extractor(
                model_id=name, drop_after=drop_after, *args, **kwargs
            )
        elif model_type == "timm":
            backbone = timm_feature_extractor(model_id=name, *args, **kwargs)
        elif model_type == "custom":
            return getattr(models.catalog, str(name))(**kwargs)
        else:
            raise ValueError(f"Invalid `model.backbone.TYPE`: `{model_type}")

        return backbone

    def setup_hook(self, network, key, layer_name, mode="output", idx=None):
        if not hasattr(self, "_hook_cache"):
            self._hook_cache = {}

        def save_to(key, mode="output", idx=None):
            def hook(m, i, output):
                # initialize array for device
                self._hook_cache[output.device.index] = self._hook_cache.get(
                    output.device.index, {}
                )
                assert mode in ("output", "input")
                if mode == "output":
                    f = output.detach()
                elif mode == "input":
                    f = i.detach()

                if idx is not None:
                    f = f[idx]
                f = f.detach()

                self._hook_cache[output.device.index][key] = f

            return hook

        layer = get_layer(network, layer_name)
        layer.register_forward_hook(save_to(key, mode=mode, idx=idx))

    def get_hook(self, key, device=None):
        device_index = device.index
        return self._hook_cache[device_index][key]

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def evaluate(self, batch, stage=None):
        raise NotImplementedError()

    def predict_step(self, batch, batch_idx):
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
        elif optimizer_name == "adamw":
            optimizer_builder = optim.AdamW
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
            elif schedule_name == "1cycle":
                schedule_builder = lr_scheduler.OneCycleLR
            elif schedule_name == "step":
                schedule_builder = lr_scheduler.StepLR
            elif schedule_name == "multi-step":
                schedule_builder = lr_scheduler.MultiStepLR
            else:
                raise ValueError(f"Invalid value for lr_scheduler: {schedule_name}")
            scheduler = schedule_builder(optimizer, **schedule_kwargs)

            if "lr_warmup" in self.training_cfg:
                warmup_cfg = self.training_cfg["lr_warmup"]
                scheduler = GradualWarmupScheduler(
                    optimizer=optimizer,
                    after_scheduler=scheduler,
                    **warmup_cfg
                )
            config["lr_scheduler"] = scheduler
        return config

    def use_ema_weights(self):
        if self.EMA is False:
            raise ValueError()
        return self.ema_manager.average_parameters()

    def _apply_ema_weights(self):
        self.ema_manager.store()
        self.ema_manager.copy_to()

    def _apply_regular_weights(self):
        self.ema_manager.restore()

    def training_epoch_end(self, outputs):
        # update callbacks
        if self.EMA:
            self.ema_manager.update()
        # log epoch-wise metrics
        total_loss = sum([x["loss"] for x in outputs])
        total_loss = total_loss / len(outputs)
        self.log("epoch/trn_loss", float(total_loss.cpu()))

    def validation_epoch_start(self):
        if self.EMA:
            self._apply_ema_weights()

    def validation_epoch_end(self, validation_step_outputs):
        # reset EMA
        if self.EMA:
            self._apply_regular_weights()

        # TODO: make it flexible for more output formats.
        total_loss, total_performance = map(sum, zip(*validation_step_outputs))
        total_performance = total_performance / len(validation_step_outputs)
        total_loss = total_loss / len(validation_step_outputs)
        self.log("epoch/val_performance", total_performance)
        self.log("epoch/val_loss", total_loss)

    def test_epoch_start(self, outputs):
        if self.EMA:
            self._apply_ema_weights()

    def test_epoch_end(self, test_step_outputs):
        # reset EMA
        if self.EMA:
            self._apply_regular_weights()

        total_loss, total_performance = map(sum, zip(*test_step_outputs))
        total_performance = total_performance / len(test_step_outputs)
        total_loss = total_loss / len(test_step_outputs)
        self.log("epoch/test_performance", total_performance)
        self.log("epoch/test_loss", total_loss)
