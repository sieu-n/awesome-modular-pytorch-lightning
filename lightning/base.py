from collections import OrderedDict
import pytorch_lightning as pl
import torchmetrics
import wandb
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn, optim
from torch.optim import lr_scheduler

from algorithms import loss as LossPool
from algorithms.optimizers.lr_scheduler.warmup import GradualWarmupScheduler
from algorithms.optimizers.sam import SAM
from models import catalog as ModelPool
from models import heads as HeadPool
from models.vision.backbone.timm import timm_feature_extractor
from models.vision.backbone.torchvision import torchvision_feature_extractor
from utils.models import get_layer
from utils.pretrained import load_model_weights

""" Build components for the lightningmodule
"""


def build_backbone(name, model_type="custom", drop_after=None, *args, **kwargs):
    if model_type == "torchvision":
        backbone = torchvision_feature_extractor(
            model_id=name, drop_after=drop_after, *args, **kwargs
        )
    elif model_type == "timm":
        backbone = timm_feature_extractor(model_id=name, *args, **kwargs)
    elif model_type == "custom":
        return getattr(ModelPool, str(name))(**kwargs)
    else:
        raise ValueError(f"Invalid `model.backbone.TYPE`: `{model_type}")

    return backbone


def build_metric(metric_type, file=None, *args, **kwargs):
    # build and return any nn.Module that is defined under `module_locations`.
    metric_pool = OrderedDict(
        {
            "torchmetrics": torchmetrics,
        }
    )
    # if library is specified
    if file:
        if type(metric_type) == str:
            metric_type = getattr(metric_pool[file], metric_type)
        else:
            raise ValueError(
                "If `file` is specified, provide the name of the metric_pool as a string."
            )
    elif type(metric_type) == str:
        is_found = False
        for location in metric_pool.values():
            if hasattr(location, metric_type):
                print(f"'{metric_type}' was found in `{location}.")
                metric_type = getattr(location, metric_type)
                is_found = True
                break
        if not is_found:
            raise ValueError(
                f"{metric_type} was not found in the pool of modules: {list(module_pool.values())}"
            )
    print(f"Building metric: '{metric_type}'")
    metric = metric_type(*args, **kwargs)
    return metric


class _BaseLightningTrainer(pl.LightningModule):
    def __init__(self, model_cfg, training_cfg, const_cfg={}, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # disable automatic_optimization.
        if "sharpness-aware" in training_cfg:
            self.automatic_optimization = False
        # save training_cfg for defining optimizers when `configure_optimizers` is called.
        self.training_cfg = training_cfg
        self.const_cfg = const_cfg
        # build backbone
        if "backbone" in model_cfg:
            backbone_cfg = model_cfg["backbone"]
            self.backbone = build_backbone(
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
        # set metrics
        self.metrics = {"trn": [], "val": [], "test": []}
        metrics = training_cfg.get("metrics", {})
        for metric_name, metric_cfg in metrics.items():
            for subset in metric_cfg["when"].split(","):
                metric = build_metric(
                    metric_type=metric_cfg["name"],
                    file=metric_cfg.get("file", None),
                    **metric_cfg.get("args", {}),
                )
                # get log frequency
                frequency = metric_cfg.get("frequency", 1)
                if type(frequency) == dict:
                    frequency = frequency.get(subset, 1)
                metric_data = {
                    "name": metric_name,
                    "metric": metric,
                    "update_keys": metric_cfg["update"],
                    "frequency": frequency,
                    "next_log": 0,
                }

                self.metrics[subset].append(metric_data)
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

    def update_metrics(self, subset, res):
        for metric_data in self.metrics[subset]:
            if metric_data["next_log"] > 0:
                continue
            update_kwargs = {
                key: res[val].cpu() for key, val in metric_data["update_keys"].items()
            }
            metric_data["metric"].update(**update_kwargs)

    def digest_metrics(self, subset):
        for metric_data in self.metrics[subset]:
            if metric_data["next_log"] > 0:
                continue
            # skip `frequency` - 1 epochs before logging.
            metric_data["next_log"] = metric_data["frequency"] - 1

            metric_name = metric_data["name"]
            res = metric_data["metric"].compute()
            metric_data["metric"].reset()

            log_key = f"epoch_{subset}/{metric_name}"
            # special metrics
            if metric_name == "confusion_matrix":
                if wandb.run is not None:
                    disp = ConfusionMatrixDisplay(
                        confusion_matrix=res.numpy(),
                        display_labels=self.const_cfg.get("label_map", None),
                    )
                    disp.plot()
                    wandb.log({log_key: disp.figure_})
                else:
                    self.log(log_key, res)
            else:
                self.log(log_key, res)

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

    def build_module(self, module_type, file=None, *args, **kwargs):
        # build and return any nn.Module that is defined under `module_locations`.
        module_pool = OrderedDict(
            {
                "heads": HeadPool,
                "loss": LossPool,
                "torch.nn": nn,
            }
        )
        # if library is specified
        if file:
            if type(module_type) == str:
                module_type = getattr(module_pool[file], module_type)
            else:
                raise ValueError(
                    "If `to_look_at` is specified, provide the name of the module as a string."
                )
        elif type(module_type) == str:
            is_found = False
            for location in module_pool.values():
                if hasattr(location, module_type):
                    print(f"'{module_type}' was found in `{location}.")
                    module_type = getattr(location, module_type)
                    is_found = True
                    break
            if not is_found:
                raise ValueError(
                    f"{module_type} was not found in the pool of modules: {list(module_pool.values())}"
                )
        head = module_type(*args, **kwargs)
        return head

    def _training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def evaluate(self, batch, stage=None):
        raise NotImplementedError()

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError()

    def manual_optimization_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        if "sharpness-aware" in self.training_cfg:
            # TODO refactoring based on `step`
            optimizer.first_step(zero_grad=True)

            loss_2, _ = self._training_step(batch, batch_idx)
            self.manual_backward(loss_2)
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()

    def training_step(self, batch, batch_idx):
        if self.automatic_optimization is False:
            optimizer = self.optimizers()
            optimizer.zero_grad()

        loss, res = self._training_step(batch, batch_idx)
        self.manual_backward(loss)

        if self.automatic_optimization is False:
            self.manual_optimization_step(batch, batch_idx)

        self.update_metrics("trn", res)
        return loss

    def validation_step(self, batch, batch_idx):
        res = self.evaluate(batch, "val")
        self.update_metrics("val", res)
        return res

    def test_step(self, batch, batch_idx):
        res = self.evaluate(batch, "test")
        self.update_metrics("test", res)
        return res

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
        # apply sharpness-aware minimization
        if "sharpness-aware" in self.training_cfg:
            sam_cfg = self.training_cfg["sharpness-aware"]
            optimizer = SAM(
                params=self.parameters(),
                base_optimizer=optimizer_builder,
                rho=sam_cfg["rho"] if "rho" in sam_cfg else 0.05,
                lr=self.training_cfg["lr"],
                **optimizer_kwargs
            )
        else:
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
            # learning rate warmup
            if "lr_warmup" in self.training_cfg:
                warmup_cfg = self.training_cfg["lr_warmup"]
                scheduler = GradualWarmupScheduler(
                    optimizer=optimizer, after_scheduler=scheduler, **warmup_cfg
                )
            config["lr_scheduler"] = scheduler
        return config

    def training_epoch_end(self, outputs):
        # log epoch-wise metrics
        self.digest_metrics("trn")

    def validation_epoch_end(self, validation_step_outputs):
        self.digest_metrics("val")

    def test_epoch_end(self, test_step_outputs):
        self.digest_metrics("test")
