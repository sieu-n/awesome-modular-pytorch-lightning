import torchvision.datasets as TD

from utils.configs import merge_config


def torchvision_dataset(name, cfg):
    """ Load dataset from torchvision."""
    subset_types = list(cfg["dataset_subset_cfg"].keys())
    print(f"[*] Attempting to load {subset_types} subsets of `{name}` dataset using `torchvision`.")
    ds_builder = getattr(TD, name)

    datasets = {}
    for subset_key in subset_types:
        incoming_cfg = cfg["dataset_subset_cfg"][subset_key]
        if not isinstance(incoming_cfg, dict):
            incoming_cfg = {}
        subset_cfg = merge_config(cfg["dataset_base_cfg"], incoming_cfg)
        # create dataset.
        datasets[subset_key] = ds_builder(**subset_cfg)

    return datasets
