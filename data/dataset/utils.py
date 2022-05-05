import torchvision.datasets as TD

from utils.configs import merge_config


def torchvision_dataset(name, config):
    """ Load dataset from torchvision."""
    subset_types = list(config["dataset_subset_config"].keys())
    print(f"[*] Attempting to load {subset_types} subsets of `{name}` dataset using `torchvision`.")
    ds_builder = TD.__dict__[name]

    datasets = {}
    for subset_key in subset_types:
        incoming_cfg = config["dataset_subset_config"][subset_key]
        if not isinstance(incoming_cfg, dict):
            incoming_cfg = {}
        subset_cfg = merge_config(config["dataset_base_config"], incoming_cfg)
        # create dataset.
        datasets[subset_key] = ds_builder(**subset_cfg)

    return datasets
