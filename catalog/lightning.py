# lightningmodule catalog.
from lightning.vision import (  # noqa
    ClassificationTrainer,
    DistillationTrainer,
    MMDetectionTrainer,
)
from utils.pretrained import load_model_weights

from ._get import _get


def get(name):
    return _get(globals(), name, "LightningModule")


def build(name, *args, **kwargs):
    return get(name)(*args, **kwargs)


def build_from_cfg(model_cfg, training_cfg, const_cfg={}):
    lightning_module = get(training_cfg["ID"])
    model = lightning_module(model_cfg, training_cfg, const_cfg)

    # load model from path if specified.
    if "state_dict_path" in model_cfg:
        load_model_weights(
            model=model,
            state_dict_path=model_cfg["state_dict_path"],
            is_ckpt=model_cfg.get("is_ckpt", False),
        )

    return model
