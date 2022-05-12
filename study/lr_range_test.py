# https://arxiv.org/pdf/1506.01186.pdf
import random
from argparse import ArgumentParser
from copy import deepcopy

from main import Experiment
from torch.utils.data.sampler import SubsetRandomSampler
from utils.configs import read_configs

if __name__ == "__main__":
    raise NotImplementedError()
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

    args = parser.parse_args()
    cfg = read_configs(args.configs)

    # setup dataset size experiment
    assert (
        hasattr(args, "range")
        + hasattr(args, "range_percent")
        + hasattr(args, "size_at_cycle")
        + hasattr(args, "size_at_cycle_percent")
    ) == 1, "one of `range`, `range_percent`, `size_at_cycle`, or \
                                            `size_at_cycle_percent` must be specified"
    train_samples = cfg["dataset"]["trn_size"]

    if args.range:
        init_samples, step = int(args.range[0]), int(args.range[1])
        data_size_cycle = range(init_samples, train_samples, step)
    elif args.range_percent:
        init_samples = int(float(args.range_percent[0]) * train_samples)
        step = int(float(args.range_percent[1]) * train_samples)
        data_size_cycle = range(init_samples, train_samples, step)
    elif args.size_at_cycle:
        data_size_cycle = [int(n) for n in args.size_at_cycle]
    elif args.size_at_cycle_percent:
        data_size_cycle = [
            (float(n) * train_samples) for n in args.size_at_cycle_percent
        ]

    if args.seed:
        random_sampler = random.Random(args.seed).sample
    else:
        random_sampler = random.sample
    # train
    experiment = Experiment(cfg)
    for idx, dataset_size in enumerate(data_size_cycle):
        print(
            f"Cycle # {idx} / {len(data_size_cycle)} | Training data size: {dataset_size}"
        )
        cycle_cfg = deepcopy(cfg)
        idx_labeled = random_sampler(range(train_samples), dataset_size)

        # control dataset size
        assert (
            "sampler" not in cycle_cfg["dataloader"]["trn"]
        ), "WARNING! try another way to control dataset size"
        cycle_cfg["dataloader"]["trn"]["sampler"] = SubsetRandomSampler(idx_labeled)

        # control logging
        cycle_cfg["name"] = f"{cycle_cfg['name']}-cycle_{idx}-{dataset_size}_samples"

        experiment.setup_experiment_from_cfg(cycle_cfg)
        result = experiment.train()
        print("Result:", result)
