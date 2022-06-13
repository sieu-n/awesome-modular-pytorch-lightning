import math
import random
from argparse import ArgumentParser
from copy import deepcopy

from main import Experiment
from torch.utils.data import Dataset
from utils.configs import read_configs
from utils.logging import log_to_wandb


class SubsetDataset(Dataset):
    def __init__(self, base_dataset, indices):
        """
        Build dataset that simply adds more data transformations to the original samples.
        Parameters
        ----------
        base_dataset: torch.utils.data.Dataset
            base dataset that is used to get source samples.
        """
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        return self.base_dataset[idx]


if __name__ == "__main__":
    # read config yaml paths
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", required=True)

    parser.add_argument("--seed", type=int, default=None, help="random seed")

    parser.add_argument("-r", "--range", nargs=2, help="seed_samples, addendum")
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

    # setup dataset size experiment
    assert (
        (args.range is None)
        + (args.range_percent is None)
        + (args.size_at_cycle is None)
        + (args.size_at_cycle_percent is None)
    ) == 3, "one of `range`, `range_percent`, `size_at_cycle`, or \
            `size_at_cycle_percent` must be specified"
    num_train_samples = cfg["dataset"]["trn_size"]

    if args.range:
        init_samples, step = int(args.range[0]), int(args.range[1])
        data_size_cycle = list(range(init_samples, num_train_samples + 1, step))
    elif args.range_percent:
        init_samples = int(float(args.range_percent[0]) * num_train_samples)
        step = int(float(args.range_percent[1]) * num_train_samples)
        data_size_cycle = list(range(init_samples, num_train_samples + 1, step))
    elif args.size_at_cycle:
        data_size_cycle = [int(n) for n in args.size_at_cycle]
    elif args.size_at_cycle_percent:
        data_size_cycle = [
            int(float(n) * num_train_samples) for n in args.size_at_cycle_percent
        ]

    if args.seed:
        random_sampler = random.Random(args.seed).sample
    else:
        random_sampler = random.sample

    experiment = Experiment(cfg)
    results = []
    for idx, dataset_size in enumerate(data_size_cycle):
        print(
            f"Cycle # {idx} / {len(data_size_cycle)} | Training data size: {dataset_size}"
        )
        cycle_cfg = deepcopy(
            cfg
        )  # copy and edit config file for dataset size experiment.
        idx_labeled = random_sampler(range(num_train_samples), dataset_size)

        # control dataset size
        assert (
            "sampler" not in cycle_cfg["dataloader"]["trn"]
        ), "try another way to control dataset size"
        cycle_cfg["dataset"]["transformations"].append(
            [
                "trn",
                [
                    {
                        "name": SubsetDataset,
                        "args": {"indices": idx_labeled},
                    }
                ],
            ]
        )
        cycle_cfg["dataset"]["trn_size"] = len(idx_labeled)

        # control logging
        if "wandb" in cfg:
            cycle_cfg["wandb"]["group"] = cfg["name"]
        cycle_cfg["name"] = f"{cycle_cfg['name']}-cycle_{idx}-{dataset_size}_samples"

        # compute number of epochs to compensate smaller number of steps.
        epochs = cfg["training"]["epochs"]
        if args.same_steps:
            epochs = math.floor(epochs * (data_size_cycle[-1]) / dataset_size)
            print(f"Increasing training epoch: {cfg['training']['epochs']} -> {epochs}")
        cfg["training"]["epochs"] = epochs

        experiment.setup_experiment_from_cfg(cycle_cfg)
        result = experiment.train(trainer_cfg=cycle_cfg["trainer"], epochs=epochs)
        print("Result:", result)
        results.append(result[0])
    print("Final results:", results)
    if "wandb" in cfg:
        log_to_wandb(
            results,
            exp_name=f"dataset-size-experiment-{cfg['name']}",
            group=cfg["wandb"].get("group", None),
            project=cfg["wandb"].get("project", None),
        )
