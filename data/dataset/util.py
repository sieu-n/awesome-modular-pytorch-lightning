from copy import deepcopy

import torchvision.datasets as TD


def merge_config(cfg_base, cfg_from):
    def recursively_write(base, f):
        for k in f.keys():
            if isinstance(f[k], dict):
                if k in base:
                    base[k] = recursively_write(deepcopy(base[k]), f[k])
                else:
                    base[k] = f[k]
            # overwrite, like the case of "optimizer/lr" and "d" in the example.
            else:
                base[k] = f[k]
        return base

    return recursively_write(deepcopy(cfg_base), cfg_from)


def torchvision_dataset(name, cfg):
    """Load dataset from torchvision."""
    subset_types = list(cfg["dataset_subset_cfg"].keys())
    print(
        f"[*] Attempting to load {subset_types} subsets of `{name}` dataset using `torchvision`."
    )
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
