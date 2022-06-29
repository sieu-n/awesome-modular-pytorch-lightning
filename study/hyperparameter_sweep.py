from email.policy import default
import math
import os
import random
from argparse import ArgumentParser
from copy import deepcopy

import pytorch_lightning as pl
from main import Experiment
from torch.utils.data import Dataset
from utils.configs import read_configs, merge_config
from utils.logging import log_to_wandb


class SubsetDataset(Dataset):
    def __init__(self, base_dataset, indices=None, size=None, seed=None):
        """
        Build dataset that simply adds more data transformations to the original samples.
        Parameters
        ----------
        base_dataset: torch.utils.data.Dataset
            base dataset that is used to get source samples.
        """
        self.base_dataset = base_dataset

        if indices is not None:
            self.indices = indices
        else:
            if args.seed:
                random_sampler = random.Random(args.seed).sample
            else:
                random_sampler = random.sample
            self.indices = random_sampler(range(len(base_dataset)), size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        return self.base_dataset[idx]


def set_key_to(d, key, value):
    # For example, given key="training.optimizer.lr" and value="0.01",
    # create an empty dictionary: d_to_push = { "training": { "optimizer": { } } }

    d_to_push = {}
    _d_to_push = d_to_push
    for k in key.split(".")[:-1]:
        _d_to_push[k] = {}
        _d_to_push = _d_to_push[k]
    # fill in 0.01: d_to_push = { "training": { "optimizer": { "lr": 0.01 } } }
    _d_to_push[key.split(".")[-1]] = value

    return merge_config(cfg_base=d, cfg_from=d_to_push)


def get_data_size_schedule(args, num_train_samples):
    if args.range:
        init_samples, step = int(args.range[0]), int(args.range[1])
        if len(args.range) == 2:
            stop = num_train_samples
        elif len(args.range) == 3:
            stop = int(args.range[2])
        else:
            raise ValueError(f"Invalid range value: {args.range}")
        data_size_schedule = list(range(init_samples, stop + 1, step))
    elif args.range_percent:
        init_samples = int(float(args.range_percent[0]) * num_train_samples)
        step = int(float(args.range_percent[1]) * num_train_samples)
        data_size_schedule = list(range(init_samples, num_train_samples + 1, step))
    elif args.size_at_cycle:
        data_size_schedule = [int(n) for n in args.size_at_cycle]
    elif args.size_at_cycle_percent:
        data_size_schedule = [
            int(float(n) * num_train_samples) for n in args.size_at_cycle_percent
        ]
    return data_size_schedule


if __name__ == "__main__":
    # read config yaml paths
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--offline", action="store_true", default=False)
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--set_same_group", default=False, action="store_true")
    parser.add_argument(
        "-v", "--value", nargs="+", help="list of values"
    )
    parser.add_argument(
        "-k", "--key", type=str
    )
    parser.add_argument(
        "-t", "--dtype", type="str", choices=["str", "int", "float"]
    )
    args = parser.parse_args()
    cfg = read_configs(args.configs)

    if args.name is not None:
        cfg["name"] = args.name
    if args.group:
        cfg["wandb"]["group"] = args.group
    if args.offline:
        cfg["wandb"]["offline"] = True

    # set hparam sweep schedule
    dtype_map = {
        "int": int,
        "float": float,
        "str": str,
    }
    values = list(map(args.value, dtype_map[args.dtype]))
    print("Hparam schedule: " + str(values))

    results = []
    for idx, value in enumerate(values):
        print(
            f"Cycle # {idx} / {len(values)} | {args.key}: {value}"
        )
        cycle_cfg = deepcopy(cfg)
        cycle_cfg["name"] = f"{cycle_cfg['name']}-cycle_{idx}-{args.key}_{value}"

        # set `args.key` in config to `value`
        set_key_to(cycle_cfg, args.key, value)
        ################################################################
        # build experiment
        ################################################################
        experiment = Experiment(cfg)
        experiment.initialize_environment(cfg=cfg)
        if "wandb" in cfg and args.set_same_group:
            cycle_cfg["wandb"]["group"] = experiment.experiment_name

        datasets = experiment.setup_dataset(
            dataset_cfg=cfg["dataset"],
            transform_cfg=cfg["transform"],
        )
        dataloaders = experiment.setup_dataloader(
            datasets=datasets,
            dataloader_cfg=cfg["dataloader"],
        )
        train_dataloader, val_dataloader = dataloaders["trn"], dataloaders["val"]
        # build model and callbacks
        model = experiment.setup_model(
            model_cfg=cfg["model"], training_cfg=cfg["training"]
        )
        logger_and_callbacks = experiment.setup_callbacks(cfg=cycle_cfg)

        ################################################################
        # train
        ################################################################
        save_path = ("checkpoints/model_state_dict.pth",)
        if not args.root_dir:
            root_dir = os.path.join(
                f"{experiment.exp_dir}/checkpoints", experiment.experiment_name
            )
        else:
            root_dir = os.path.join(args.root_dir, experiment.experiment_name)
        epochs = cfg["training"]["epochs"]
        # lightning trainer
        pl_trainer = pl.Trainer(
            max_epochs=epochs,
            default_root_dir=root_dir,
            **(
                logger_and_callbacks
                if hasattr(experiment, "logger_and_callbacks")
                else {}
            ),
            **cfg["trainer"],
        )
        # train
        pl_trainer.fit(
            model,
            train_dataloader,
            val_dataloader,
        )
        # log results
        if (
            hasattr(experiment, "logger_and_callbacks")
            and "logger" in logger_and_callbacks
        ):
            logger_and_callbacks["logger"].experiment.finish()
        # test
        res = pl_trainer.test(model, val_dataloader)
        res[0][args.key] = value
        print("Result:", res)
        results.append(res[0])

    print("Final results:", results)
    if "wandb" in cfg:
        log_to_wandb(
            results,
            exp_name=f"dataset-size-experiment-{cfg['name']}",
            group=cfg["wandb"].get("group", None),
            project=cfg["wandb"].get("project", None),
        )
