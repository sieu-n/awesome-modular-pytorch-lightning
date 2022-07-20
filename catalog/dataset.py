# dataset catalog
import torchvision.datasets as TD
try:
    from mmdet.datasets import build_dataset as build_dataset_mmdet
except ImportError:
    pass

# utils
from ._get import _get


def get(name):
    return _get(globals(), name, "Dataset")


def build_dataset(dataset_type, name=None, cfg=None, *args, **kwargs):
    # returns: dict{subset_key: torch.utils.data.Dataset, ...}
    if dataset_type == "torchvision":
        ds_builder = getattr(TD, name)
        print(
            f"[*] Attempting to build `{ds_builder}`."
        )
        return ds_builder(*args, **kwargs)
    elif dataset_type == "mmdetection":
        print(
            f"[*] Attempting to load {cfg['type']}dataset from `mmdetection`."
        )
        return build_dataset_mmdet(cfg, *args, **kwargs)
    else:
        raise ValueError(f"Invalid dataset type: `{dataset_type}`")
