import os
from pathlib import Path
from glob import glob

import torch
import torchvision
import torchvision.datasets as TD

from utils.configs import merge_config

def torchvision_dataset(name, config, get_val=True):
    
    datasets = [] 
    for subset_key in config["dataset_subset_config"].keys(): 
        new_cfg = merge_config(config["dataset_base_config"], config["dataset_subset_config"][subset_key])

    ds = TD.__dict__[name]

    return ds(**config)
