from argparse import ArgumentParser
from utils.experiment import read_configs
from main import Experiment


if __name__ == "__main__":
    # read config yaml paths
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", required=True)

    parser.add_argument("-r", "--range", nargs=2, help="seed_samples, addendum")
    parser.add_argument("-p", "--range_percent", nargs=2, help="seed_samples_percent, addendum_percent")

    args = parser.parse_args()
    cfg = read_configs(args.configs)

    # setup dataset size experiment
    assert (hasattr(args, "range") + hasattr(args, "range_percent")) == 1, "one of `rang` or `range_percent` must be \
                                                                            specified"
    train_samples = cfg["dataset"]["trn_size"]

    if args.range:
        init_samples, step = int(args.range[0]), int(args.range[1])
        data_size_cycle = range(init_samples, train_samples, step)
    elif args.range_percent:
        init_samples = int(float(args.range_percent[0]) * train_samples)
        step = int(float(args.range_percent[1]) * train_samples)
        data_size_cycle = range(init_samples, train_samples, step)

    # train
    experiment = Experiment(cfg)
    for idx, dataset_size in enumerate(data_size_cycle):
        print(f"Cycle # {idx} / {len(data_size_cycle)} | Training data size: {dataset_size}")
        experiment.setup_experiment_from_cfg(cfg)
        
        result = experiment.train()
        print("Result:", result)
