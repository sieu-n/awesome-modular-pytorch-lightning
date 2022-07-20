import catalog.metric
import catalog.models
import catalog.modules
import catalog.TTA_modules
import wandb
from sklearn.metrics import ConfusionMatrixDisplay
from utils.models import get_layer
from utils.pretrained import load_model_weights

from .common import _LightningModule


class _BaseLightningTrainer(_LightningModule):
    def __init__(self, model_cfg, training_cfg, const_cfg={}, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        print("[*] Building model components")
        self.const_cfg = const_cfg
        self.training_cfg = training_cfg
        # disable automatic_optimization.
        if "sharpness-aware" in training_cfg:
            self.automatic_optimization = False
            print(
                "Automatic optimization feature of pytorch-lighining is disabled because of `sharpness-aware-minimization`. \
                   Be aware of unexpected behavior regarding custom learning rate schedule and optimizers."
            )
        # build backbone
        if "backbone" in model_cfg:
            backbone_cfg = model_cfg["backbone"]
            print(f"(1/4) Building backbone model: {backbone_cfg['ID']}")

            self.backbone = catalog.models.build_backbone(
                name=backbone_cfg["ID"],
                model_type=backbone_cfg["TYPE"],
                drop_after=backbone_cfg.get("drop_after", None),
                **backbone_cfg.get("cfg", {}),
            )
            # load backbone weights from url / filepath
            if "weights" in backbone_cfg:
                print(f"Using pretrained backbone: {backbone_cfg['weights']}")
                self.backbone = load_model_weights(
                    model=self.backbone, **backbone_cfg["weights"]
                )
        # build modules
        if "modules" in model_cfg:
            print("(2/4) Building modules attached to the backbone model...")
            modules = model_cfg["modules"]
            for module_name, module_cfg in modules.items():
                head_module = catalog.modules.build(
                    name=module_cfg["name"],
                    file=module_cfg.get("file", None),
                    **module_cfg.get("args", {}),
                )
                setattr(self, module_name, head_module)
        # set metrics
        if "metrics" in model_cfg:
            print("(3/4) Building metrics:")
            self.metrics = {"trn": [], "val": [], "test": []}
            metrics = training_cfg["metrics"]
            for metric_name, metric_cfg in metrics.items():
                subsets_to_compute = metric_cfg.get("when", "val")
                for subset in subsets_to_compute.split(","):
                    metric = catalog.metric.build(
                        name=metric_cfg["name"],
                        file=metric_cfg.get("file", None),
                        **metric_cfg.get("args", {}),
                    )
                    # get log frequency
                    interval = metric_cfg.get("interval", 1)
                    if type(interval) == dict:
                        interval = interval.get(subset, 1)
                    metric_data = {
                        "name": metric_name,
                        "metric": metric,
                        "update_keys": metric_cfg["update"],
                        "interval": interval,
                        "next_log": 0,
                    }
                    self.metrics[subset].append(metric_data)
        # setup hooks
        self._hook_cache = {}
        if "hooks" in model_cfg:
            print("(4/4) Setting up hooks")
            hooks = model_cfg["hooks"]
            for hook_name, hook_cfg in hooks.items():
                self.setup_hook(
                    network=getattr(self, hook_cfg["model"]),
                    key=hook_name,
                    layer_name=hook_cfg["layer_name"],
                    **hook_cfg.get("cfg", {}),
                )
        # tta
        self.is_tta_enabled = False
        if "tta" in model_cfg:
            tta_cfg = model_cfg["tta"]
            self.is_tta_enabled = True
            self.TTA_module = catalog.TTA_modules.build(
                name=tta_cfg["name"],
                model=self,
                training_cfg=training_cfg,
                const_cfg=const_cfg,
                **tta_cfg["args"],
            )

    def enable_tta(self, TTA_module=None):
        if TTA_module is None:
            assert hasattr(
                self, "TTA_module"
            ), "TTA_module is not assigned nor provided."
        else:
            # assign new tta module
            self.TTA_module = TTA_module

        self.is_tta_enabled = True

    def disable_tta(self):
        self.is_tta_enabled = False

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
            # log every `interval` epochs.
            if metric_data["next_log"] > 0:
                metric_data["next_log"] -= 1
                continue
            # skip `interval` - 1 times.
            metric_data["next_log"] = metric_data["interval"] - 1

            metric_name = metric_data["name"]
            res = metric_data["metric"].compute()
            metric_data["metric"].reset()

            log_key = f"epoch_{subset}/{metric_name}"
            # special metrics with specific names are treated differently.
            if metric_name == "confusion_matrix":
                if wandb.run is not None:
                    # log confusion matrix as image to wandb.
                    disp = ConfusionMatrixDisplay(
                        confusion_matrix=res.numpy(),
                        display_labels=self.const_cfg.get("label_map", None),
                    )
                    disp.plot()
                    wandb.log({log_key: disp.figure_})
                else:
                    self.log(log_key, res)
            else:
                # typical metrics
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

    def _training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def evaluate(self, batch, stage=None):
        raise NotImplementedError()

    def _predict_step(self, batch, batch_idx):
        raise NotImplementedError()

    def manual_optimization_step(self, loss, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        if "sharpness-aware" in self.training_cfg:
            # TODO refactoring based on `step`
            optimizer.first_step(zero_grad=True)

            loss_2, _ = self._training_step(batch, batch_idx)
            self.manual_backward(loss_2)
            if "gradient_clip_val" in self.training_cfg:
                self.clip_gradients(
                    optimizer, gradient_clip_val=self.training_cfg["gradient_clip_val"]
                )

            optimizer.second_step(zero_grad=True)
        else:
            if "gradient_clip_val" in self.training_cfg:
                self.clip_gradients(
                    optimizer, gradient_clip_val=self.training_cfg["gradient_clip_val"]
                )
            optimizer.step()

        # custom lr scheduler step
        for sch in self.trainer.lr_scheduler_configs:
            if sch.interval == "step":
                sch.scheduler.step()

    def training_step(self, batch, batch_idx):
        loss, res = self._training_step(batch, batch_idx)

        if self.automatic_optimization is False:
            self.manual_optimization_step(loss, batch, batch_idx)

        self.update_metrics("trn", res)
        return loss

    def validation_step(self, batch, batch_idx):
        res = self.evaluate(batch, "val")
        self.update_metrics("val", res)
        return res

    def test_step(self, batch, batch_idx):
        if self.is_tta_enabled:
            # apply test-time augmentation if specified.
            pred, res = self.TTA_module(batch)
        else:
            res = self.evaluate(batch, "test")
        self.update_metrics("test", res)
        return res

    def predict_step(self, batch, batch_idx):
        if self.is_tta_enabled:
            # apply test-time augmentation if specified.
            pred, res = self.TTA_module(batch)
        else:
            pred = self._predict_step(batch, batch_idx)
        return pred

    def training_epoch_end(self, outputs):
        # log epoch-wise metrics
        self.digest_metrics("trn")

        if self.automatic_optimization is False:
            # custom lr scheduler step
            for sch in self.trainer.lr_scheduler_configs:
                if sch.interval == "epoch":
                    sch.scheduler.step()

    def validation_epoch_end(self, validation_step_outputs):
        self.digest_metrics("val")

    def test_epoch_end(self, test_step_outputs):
        self.digest_metrics("test")
