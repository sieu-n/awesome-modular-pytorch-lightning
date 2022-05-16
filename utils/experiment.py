import json
import os
import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import data.transforms.vision as DT_V
import models.model as TorchModel
import yaml
from data.dataset.util import torchvision_dataset
from data.transforms.common import ApplyDataTransformations, ComposeTransforms

from .verbose import set_verbose

""" Implement utilities used in `main.py`.
"""


def build_network(model_cfg):
    if model_cfg["TYPE"] == "custom":
        model = getattr(TorchModel, model_cfg["ID"])(model_cfg)
    elif model_cfg["TYPE"] == "pretrained":
        raise NotImplementedError()
    else:
        raise ValueError(f"Invalid `model.TYPE`: `{model_cfg['TYPE']}")

    return model


def build_dataset(dataset_cfg, transform_cfg, const_cfg):
    # 1. build initial dataset to read data.
    dataset_mode = dataset_cfg["MODE"]
    if dataset_mode == "torchvision":
        datasets = torchvision_dataset(dataset_cfg["NAME"], dataset_cfg)
    elif dataset_mode == "from-directory":
        raise NotImplementedError("TODO!")
    else:
        raise ValueError(f"Invalid dataset type: `{dataset_mode}`")
    # datasets: dict{subset_key: torch.utils.data.Dataset, ...}

    # 2.1. build list of transformations using `transform` defined in config.
    # transforms: dict{subset_key: [t1, t2, ...], ...}
    TRANSFORM_DECLARATIONS = [DT_V]
    transforms = {subset: [] for subset in datasets.keys()}
    for subsets, t_configs in transform_cfg:
        t = []
        # for each element of transforms,
        for t_config in t_configs:
            name, kwargs = t_config["name"], t_config.get("args", {})
            if type(name) == str:
                # find transform name that matches `name` from TRANSFORM_DECLARATIONS
                is_name_in = [hasattr(file, name) for file in TRANSFORM_DECLARATIONS]
                assert (
                    sum(is_name_in) == 1
                ), f"Transform `{name}` was found in `{sum(is_name_in)} files."
                file = TRANSFORM_DECLARATIONS[is_name_in.index(True)]
                print(
                    f"Transform {name} --> {getattr(file, name)}: found in {file.__name__}"
                )
                name = getattr(file, name)

            # build transform using arguments.
            kwargs["const_cfg"] = const_cfg     # feed const data such as label map.
            t.append(name(**kwargs))

        for subset in subsets.split(","):
            transforms[subset] += t

    # 2.2. actually apply transformations.
    transforms = {
        subset: ComposeTransforms(transforms[subset]) for subset in transforms.keys()
    }
    datasets = {
        subset: ApplyDataTransformations(
            base_dataset=datasets[subset], transforms=transforms[subset]
        )
        for subset in datasets.keys()
    }
    # 3. apply datasets.
    DATASET_DECLARATIONS = []
    apply_dataset_cfg = dataset_cfg["transformations"]
    for subsets, d_configs in apply_dataset_cfg:
        d_operations = []
        # for each element of transforms,
        for d_config in d_configs:
            name, kwargs = d_config["name"], d_config["args"]
            if type(name) == str:
                # find transform name that matches `name` from TRANSFORM_DECLARATIONS
                is_name_in = [hasattr(file, name) for file in DATASET_DECLARATIONS]
                assert (
                    sum(is_name_in) == 1
                ), f"Dataset `{name}` was found in `{sum(is_name_in)} files."
                file = DATASET_DECLARATIONS[is_name_in.index(True)]
                print(
                    f"Dataset {name} --> {getattr(file, name)}: found in {file.__name__}"
                )
                name = getattr(file, name)

            # build dataset operation using arguments.
            d_operations.append(lambda base_dataset: name(base_dataset, **kwargs))

        for subset in subsets.split(","):
            for d_operation in d_operations:
                datasets[subset] = d_operation(datasets[subset])
    return datasets


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

    # set os.environ
    set_verbose(verbose)
    timestamp = get_timestamp()
    experiment_name = f"{base_name}-{timestamp}"
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

        filename = f"configs/logs/{experiment_name}"

        # pkl should be guaranteed to work.
        print(f"Saving config to: {filename}.pkl")
        with open(filename + ".pkl", "wb") as file:
            pickle.dump(cfg, file)

        print(f"Saving config to: {filename}.yaml")
        with open(filename + ".yaml", "w") as file:
            yaml.dump(pretty_cfg, file, allow_unicode=True, default_flow_style=False)

        print(f"Saving config to: {filename}.json")
        with open(filename + ".json", "w") as file:
            json.dump(pretty_cfg, file)

    print_to_end("=")
    return experiment_name


def makedir(path):
    # check if `path` is file and remove last component.
    if Path(path).stem != path.split("/")[-1]:
        path = Path(path).parent
    if not os.path.exists(path):
        os.makedirs(path)


def get_timestamp():
    return datetime.now().strftime("%b%d_%H-%M-%S")


def print_to_end(char="#"):
    rows, columns = os.popen("stty size", "r").read().split()
    columns = max(int(columns), 40)
    spaces = char * (columns // len(char))
    print(spaces)
