from argparse import ArgumentParser
from utils.configs import read_configs
from main import Experiment
import random
from torch.utils.data import Dataset
from copy import deepcopy


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
    parser.add_argument("-rp", "--range_percent", nargs=2, help="seed_samples_percent, addendum_percent")
    parser.add_argument("-s", "--size_at_cycle", nargs="+", help="dataset size at each cycle")
    parser.add_argument("-sp", "--size_at_cycle_percent", nargs="+", help="dataset size at each cycle percent")

    args = parser.parse_args()
    cfg = read_configs(args.configs)

    # setup dataset size experiment
    assert ((args.range is None) + (args.range_percent is None) + (args.size_at_cycle is None)
            + (args.size_at_cycle_percent is None)) == 3, "one of `range`, `range_percent`, `size_at_cycle`, or \
            `size_at_cycle_percent` must be specified"
    num_rain_samples = cfg["dataset"]["trn_size"]

    if args.range:
        init_samples, step = int(args.range[0]), int(args.range[1])
        data_size_cycle = range(init_samples, num_rain_samples, step)
    elif args.range_percent:
        init_samples = int(float(args.range_percent[0]) * num_rain_samples)
        step = int(float(args.range_percent[1]) * num_rain_samples)
        data_size_cycle = range(init_samples, num_rain_samples, step)
    elif args.size_at_cycle:
        data_size_cycle = [int(n) for n in args.size_at_cycle]
    elif args.size_at_cycle_percent:
        data_size_cycle = [(float(n) * num_rain_samples) for n in args.size_at_cycle_percent]

    if args.seed:
        random_sampler = random.Random(args.seed).sample
    else:
        random_sampler = random.sample
    # train
    experiment = Experiment(cfg)
    results = []
    for idx, dataset_size in enumerate(data_size_cycle):
        print(f"Cycle # {idx} / {len(data_size_cycle)} | Training data size: {dataset_size}")
        cycle_cfg = deepcopy(cfg)
        idx_labeled = random_sampler(range(num_rain_samples), dataset_size)

        # control dataset size
        assert "sampler" not in cycle_cfg["dataloader"]["trn"], "WARNING! try another way to control dataset size"
        cycle_cfg["dataset"]["transformations"].append(
            "trn",
            [{
                "name": SubsetDataset,
                "args": {"indices": idx_labeled},
            }]
        )
        cycle_cfg["dataset"]["trn_size"] = len(idx_labeled)

        # control logging
        cycle_cfg["name"] = f"{cycle_cfg['name']}-cycle_{idx}-{dataset_size}_samples"

        experiment.setup_experiment_from_cfg(cycle_cfg)
        result = experiment.train(trainer_cfg=cycle_cfg["trainer"])
        print("Result:", result)
        results.append(result)
    print("Final results:", results)
