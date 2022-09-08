import json
import os
import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import warnings

import catalog
import yaml
from data.transforms.utils import ApplyTransforms, ComposeTransforms

from .verbose import set_verbose

""" Implement utilities used in `main.py`.
"""


########################################################################
# Find transforms and build dataset.
########################################################################
def apply_dataset_mapping(base_datasets, mapping_cfg, const_cfg):
    # returns: dict{subset_key: [t1, t2, ...], ...}
    for subsets, d_configs in deepcopy(mapping_cfg):
        # for each element of dataset mappings,
        for d_config in d_configs:
            # 1. build components of dataset mapping
            f_name, kwargs = d_config["name"], d_config.get("args", {})
            # find transform from name
            data_mapper = catalog.dataset_mapping.get(f_name)
            # build transform using arguments.
            kwargs["const_cfg"] = const_cfg  # feed const data such as label map.

            # 2. apply dataset mapping to each subset specified.
            for subset in subsets.split(","):
                if subset in base_datasets:
                    base_datasets[subset] = data_mapper(base_datasets[subset], **kwargs)
                else:
                    warnings.warn(f"{subset} was found in dataset mappings but no {subset} of \
                                    dataset was not provided.")

    return base_datasets


def build_transforms(transform_cfg, const_cfg):
    # returns: dict{subset_key: [t1, t2, ...], ...}
    transforms = {}
    for subsets, t_configs in deepcopy(transform_cfg):
        t = []
        # for each element of transforms,
        for t_config in t_configs:
            f_name, kwargs = t_config["name"], t_config.get("args", {})
            # find transform from name
            transform_f = catalog.transforms.get(f_name)
            # build transform using arguments.
            kwargs["const_cfg"] = const_cfg  # feed const data such as label map.
            t.append(transform_f(**kwargs))

        for subset in subsets.split(","):
            # add single transform
            transforms[subset] = transforms.get(subset, []) + t
    composed = {
        subset: ComposeTransforms(transforms[subset]) for subset in transforms.keys()
    }
    return composed


def build_initial_transform(initial_transform_cfg, const_cfg):
    initial_transform_cfg = deepcopy(initial_transform_cfg)
    # 2.2. actually apply transformations.
    initial_transform = None
    # find transform from name
    transform_f = catalog.transforms.get(initial_transform_cfg["name"])
    # build transform using arguments.
    kwargs = initial_transform_cfg.get("args", {})
    kwargs["const_cfg"] = const_cfg
    initial_transform = transform_f(**kwargs)
    return initial_transform


def apply_transforms(dataset, initial_transform=None, transforms=None):
    return ApplyTransforms(
        base_dataset=dataset,
        initial_transform=initial_transform,
        transforms=transforms,
    )


########################################################################
# Utility functions for managing filepath and environment variables.
########################################################################
def replace_non_json_serializable(cfg):
    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False

    if type(cfg) == dict:
        for key, value in cfg.items():
            if not is_jsonable(value):
                cfg[key] = replace_non_json_serializable(cfg[key])
        return cfg
    else:
        return (
            cfg
            if is_jsonable(cfg)
            else f"instance of {cfg.__class__.__name__}, pls check pkl."
        )


def initialize_environment(
    cfg=None, base_name="default-experiment", verbose="DEFAULT", debug_mode=False
):
    if cfg:
        base_name = cfg["name"]
        verbose = cfg.get("VERBOSE", "DEFAULT")
        debug_mode = "TRUE" if ("DEBUG_MODE" in cfg and cfg["DEBUG_MODE"]) else "FALSE"

    # set experiment name.
    set_verbose(verbose)
    timestamp = get_timestamp()
    experiment_name = f"{base_name}-{timestamp}"
    exp_dir = f"results/{experiment_name}"

    os.environ["DEBUG_MODE"] = debug_mode

    if cfg:
        # print final config.
        pretty_cfg = replace_non_json_serializable(deepcopy(cfg))
        pretty_cfg = json.dumps(pretty_cfg, indent=2, sort_keys=True)
        print_to_end("=")

        print("modular-PyTorch-lightning")
        print("Env setup is completed, start_time:", timestamp)
        print("")
        print("Final config after merging:", pretty_cfg)

        cfg_log_dir = f"{exp_dir}/configs"
        if not os.path.exists(cfg_log_dir):
            os.makedirs(cfg_log_dir)
        filename = f"{cfg_log_dir}/cfg"

        # pkl should be guaranteed to work.
        print(f"Saving config to: {filename}.pkl")
        with open(filename + ".pkl", "wb") as file:
            pickle.dump(cfg, file)

        print(f"Saving config to: {filename}.yaml")
        with open(filename + ".yaml", "w") as file:
            yaml.dump(cfg, file, allow_unicode=True, default_flow_style=False)

        print(f"Saving config to: {filename}.json")
        with open(filename + ".json", "w") as file:
            json.dump(cfg, file)

    return experiment_name, exp_dir


def makedir(path):
    # check if `path` is file and remove last component.
    if Path(path).stem != path.split("/")[-1]:
        path = Path(path).parent
    if not os.path.exists(path):
        os.makedirs(path)


def get_timestamp():
    return datetime.now().strftime("%b%d_%H-%M-%S")


def print_to_end(char="#", max_len=100):
    try:
        rows, columns = os.popen("stty size", "r").read().split()
    except ValueError:
        return
    columns = max(int(columns), max_len)
    spaces = char * (columns // len(char))
    print(spaces)


def print_d(d, p):  # noqa
    print(p, ":")
    if type(d) == dict:
        print("keys:", d.keys())
        for k in d:
            p.append(k)
            print_d(d[k])
            p.pop()
    else:
        try:
            print("len:", len(d))
        except:  # noqa
            print("no len")
        try:
            print("shape:", d.shape)
        except:  # noqa
            print("no shape")
        try:
            print("type:", type(d))
        except:  # noqa
            print("no type")
