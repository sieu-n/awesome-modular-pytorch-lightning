import math
import os
import random
from argparse import ArgumentParser
from copy import deepcopy

import pytorch_lightning as pl
from main import Experiment
from torch.utils.data import Dataset
from utils.configs import read_configs
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


def get_data_size_schedule(args, num_train_samples):
    assert (
        (args.range is None)
        + (args.range_percent is None)
        + (args.size_at_cycle is None)
        + (args.size_at_cycle_percent is None)
    ) == 3, "one of `range`, `range_percent`, `size_at_cycle`, or \
            `size_at_cycle_percent` must be specified"

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

    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for defining dataset."
    )
    parser.add_argument(
        "-r", "--range", nargs="+", help="(start, step) or (start, step, stop)"
    )
    parser.add_argument(
        "-rp", "--range_percent", nargs=2, help="seed_samples_percent, addendum_percent"
    )
    parser.add_argument(
        "-s", "--size_at_cycle", nargs="+", help="dataset size at each cycle"
    )
    parser.add_argument(
        "-sp",
        "--size_at_cycle_percent",
        nargs="+",
        help="dataset size at each cycle percent",
    )
    parser.add_argument(
        "--same_steps",
        default=False,
        action="store_true",
        help="Increase epochs so # optimization steps are same regardless of dataset size",
    )

    args = parser.parse_args()
    cfg = read_configs(args.configs)

    if args.name is not None:
        cfg["name"] = args.name
    if args.group:
        cfg["wandb"]["group"] = args.group
    if args.offline:
        cfg["wandb"]["offline"] = True

    experiment = Experiment(cfg)
    experiment.initialize_environment(cfg=cfg)
    datasets = experiment.setup_dataset(
        dataset_cfg=cfg["dataset"],
        transform_cfg=cfg["transform"],
    )
    trn_base_dataset, val_dataset = datasets["trn"], datasets["val"]

    val_dataloader = experiment.setup_dataloader(
        datasets=val_dataset,
        dataloader_cfg=cfg["dataloader"],
        subset_to_get="val",
    )

    # setup dataset size experiment
    data_size_schedule = get_data_size_schedule(args, len(trn_base_dataset))
    print("Data size schedule: " + str(data_size_schedule))
    results = []
    for idx, dataset_size in enumerate(data_size_schedule):
        print(
            f"Cycle # {idx} / {len(data_size_schedule)} | Training data size: {dataset_size}"
        )
        # setup environment
        cycle_cfg = deepcopy(cfg)
        cycle_cfg["name"] = f"{cycle_cfg['name']}-cycle_{idx}-{dataset_size}_samples"
        experiment.initialize_environment(cfg=cycle_cfg)
        if "wandb" in cfg:
            cycle_cfg["wandb"]["group"] = experiment.experiment_name

        # control dataset size and build train dataloader
        trn_dataset = SubsetDataset(trn_base_dataset, size=dataset_size, seed=args.seed)
        trn_dataloader = experiment.setup_dataloader(
            datasets=trn_dataset,
            dataloader_cfg=cfg["dataloader"],
            subset_to_get="trn",
        )

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
        # compute number of epochs to compensate smaller number of steps.
        epochs = cfg["training"]["epochs"]
        if args.same_steps:
            epochs = math.floor(epochs * len(trn_base_dataset) / dataset_size)
            print(f"Increasing training epoch: {cfg['training']['epochs']} -> {epochs}")
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
            trn_dataloader,
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
