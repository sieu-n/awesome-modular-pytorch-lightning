from data.dataset.utils import torchvision_dataset
from data.transforms.utils import (ApplyDataTransformations, ComposeTransforms)
import data.transforms.vision as DT_V

from .verbose import set_verbose

""" Implement utilities used in `main.py`.
"""


def build_dataset(dataset_cfg, transform_cfg):
    # 1. build initial dataset to read data.
    dataset_mode = dataset_cfg["MODE"]
    if dataset_mode == "torchvision":
        datasets = torchvision_dataset(dataset_cfg["NAME"], dataset_cfg)
    elif dataset_mode == "from-directory":
        raise NotImplementedError("TODO!")
    else:
        raise ValueError(f"Invalid dataset type: `{dataset_mode}`")
    # datasets: dict{subset_key: torch.utils.data.Dataset, ...}

    # 2. build list of transformations using `transform` defined in config.
    # transforms: dict{subset_key: [t1, t2, ...], ...}
    transforms = {subset: [] for subset in datasets.keys()}
    for subsets, t_configs in transform_cfg:
        t = []
        # for each element of transforms,
        for t_config in t_configs:
            # parse `name``: str and `kwargs`: str
            for x in t_config.items():
                name, kwargs = x
            # find transform name that matches `name` from TRANSFORM_DECLARATIONS
            TRANSFORM_DECLARATIONS = [DT_V]
            is_name_in = [name in file.__dict__ for file in TRANSFORM_DECLARATIONS]
            assert sum(is_name_in) == 1, f"Transform `{name}` was found in `{sum(is_name_in)} files."
            file = TRANSFORM_DECLARATIONS[is_name_in.index(True)]
            transform_f = file.__dict__[name]
            print(f"[*] Transform {name} --> {transform_f}: found in {file.__name__}")

            # build transform using arguments.
            t.append(transform_f(**kwargs))

        for subset in subsets.split(","):
            transforms[subset] += t

    # 3. apply transformations and return datasets that will actually be used.
    transforms = {subset: ComposeTransforms(transforms[subset])}
    return {subset: ApplyDataTransformations(base_dataset=datasets[subset], transforms=transforms[subset])
            for subset in datasets.keys()}


def setup_env(config):
    verbose = config.get("VERBOSE", default="DEFAULT")
    set_verbose(verbose)
