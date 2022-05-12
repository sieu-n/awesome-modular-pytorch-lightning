from copy import deepcopy

import yaml


def read_yaml(yaml_path):
    """Load configs from yaml file and return dictionary."""
    with open(yaml_path, "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return cfg


def merge_config(cfg_base, cfg_from):
    """
    Overwrite `cfg_base` with values of `cfg_from`. For example:
    cfg_base = {
        "optimizer": {
            "name": "adam",
            "lr": 0.001,
        },
        "d": 3,
        "e": 1
    }
    cfg_from = {
        "optimizer": {
            "lr": 0.1,
            "momentum": 0.99,
        },
        "d": 1
    }
    --> cfg_from = {
        "optimizer": {
            "name": "adam",
            "lr": 0.1,
            "momentum": 0.99,
        },
        "d": 1,
        "e": 1
    }

    Parameters
    ----------
    cfg_base: dict
        Dictionary of config to merge into.
    cfg_from: dict
        Dictionary of config to merge keys from.
    Returns
    -------
    dict
        Dictionary combining the 2 configs.
    """

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


def read_configs(yaml_paths):
    """
    Load and combine multiple yaml files and return final config.

    Parameters
    ----------
    yaml_path: list[str]
        paths to yaml file in order of importance in merging.
    Returns
    -------
    dict
        Dictionary contining final configs from multiple `yaml` files.
    """
    cfg = {}
    assert len(yaml_paths) > 0
    # override last config.
    for yaml_path in yaml_paths[::-1]:
        new_cfg = read_yaml(yaml_path)
        cfg = merge_config(cfg, new_cfg)
    return cfg
