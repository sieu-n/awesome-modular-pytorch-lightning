import catalog
import torch
import warnings

import wandb
from sklearn.metrics import ConfusionMatrixDisplay
from utils.experiment import print_to_end
from utils.hook import Hook
from utils.pretrained import load_model_weights

from .common import _LightningModule


class _BaseLightningTrainer(_LightningModule):
    def __init__(self, model_cfg, training_cfg, const_cfg={}) -> None:
        super().__init__()

        print_to_end("=")
        print_to_end("=")
        print("[*] Building model components")
        print_to_end("=")
        print_to_end("=")

        self.const_cfg = const_cfg
        self.training_cfg = training_cfg
        # disable automatic_optimization.
        if "sharpness-aware" in training_cfg:
            self.automatic_optimization = False
            print(
                "Automatic optimization feature of pytorch-lighining is disabled because of `sharpness-aware-minimization`. \
                   Be aware of unexpected behavior regarding custom learning rate schedule and optimizers."
            )
        # 1. build backbone
        if "backbone" in model_cfg:
            warnings.warn("backbone will be deprecated.", DeprecationWarning)
            backbone_cfg = model_cfg["backbone"]
            print(f"(1/6) Building backbone model: {backbone_cfg['name']}")
            self.backbone = catalog.backbone.build(
                name=backbone_cfg["name"],
                args=backbone_cfg.get("args", {}),
            )
            # load backbone weights from url / filepath
            if "weights" in backbone_cfg:
                print(f"Using pretrained backbone: {backbone_cfg['weights']}")
                self.backbone = load_model_weights(
                    model=self.backbone, **backbone_cfg["weights"]
                )
        else:
            print("(1/6) `model.backbone` is not specified. Skipping backbone model")

        # 2. build modules
        if "modules" in model_cfg:
            print("(2/6) Building modules attached to the backbone model...")
            modules = model_cfg["modules"]
            for module_name, module_cfg in modules.items():
                module = catalog.modules.build(
                    name=module_cfg["name"],
                    file=module_cfg.get("file", None),
                    **module_cfg.get("args", {}),
                )
                # load weights from url / filepath
                if "weights" in module_cfg:
                    print(f"Loading pretrained `{module_name}`: {module_cfg['weights']}")
                    module = load_model_weights(
                        model=module, **module_cfg["weights"]
                    )

                setattr(self, module_name, module)

        else:
            print("(2/6) `model.modules` is not specified. Skipping building modules")

        # 3. set metrics
        self.metrics = {"trn": [], "val": [], "test": []}
        if "metrics" in training_cfg:
            print("(3/6) Building metrics:")
            metrics = training_cfg["metrics"]
            for metric_name, metric_cfg in metrics.items():
                subsets_to_compute = metric_cfg.get("when", "val")
                for subset in subsets_to_compute.split(","):
                    metric = catalog.metric.build(
                        name=metric_cfg["name"], args=metric_cfg.get("args", {})
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
        else:
            print("(3/6) `model.metrics` is not specified. Skipping metrics")

        # 4. call overrided init function defind in child class.
        print(f"(4/6) init function of `{type(self)}`")
        self.init(model_cfg, training_cfg)

        # 5. setup feature-extraction hooks
        # TODO: implement FeatureExtractor callback instead
        if "feature_hooks" in model_cfg:
            print("(5/6) Setting up hooks for feature-extraction")
            self.hook = Hook(network=self, cfg=model_cfg["feature_hooks"])
        else:
            print("(5/6) `model.feature_hooks` is not specified.")

        # 6. setup init hook that can modify model at init. Callbacks are limited
        # as they can only be called after model is created & wrapped with
        # `pl.Trainer`. One particular use case is a hook to modify the model
        # architecture.
        if "init_hook" in model_cfg:
            print("(6/6) Calling initialization hooks.")
            init_hooks = model_cfg["init_hook"]
            for hook_name, hook_cfg in init_hooks.items():
                # find hook
                hook_f = catalog.init_hooks.get(hook_cfg["name"])
                # run hook
                hook_f(
                    self,
                    **hook_cfg.get("args", {}),
                )
        else:
            print("(6/6) `model.hooks` is not specified.")

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
        def _make_feedable(v):
            if isinstance(v, torch.Tensor):
                v = v.cpu()
            return v

        for metric_data in self.metrics[subset]:
            if metric_data["next_log"] > 0:
                continue
            update_kwargs = {
                key: _make_feedable(res[val])
                for key, val in metric_data["update_keys"].items()
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
            self.log(log_key, res, metric_name=metric_name)

    def log(self, log_key, res, metric_name=None, *args, **kwargs):
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
                super().log(log_key, res)
        else:
            # typical metrics
            if isinstance(res, dict):
                for k in list(res.keys()):
                    res[f"{log_key}/{k}"] = res.pop(k)
                self.log_dict(res)
            else:
                super().log(log_key, res)

    def init(self):
        raise NotImplementedError()

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
