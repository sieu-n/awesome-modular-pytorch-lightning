from argparse import ArgumentParser
from utils.configs import read_configs
from main import Experiment


if __name__ == "__main__":
    # read config yaml paths
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", required=True)

    args = parser.parse_args()
    cfg = read_configs(args.configs)

    # train
    experiment = Experiment(cfg)
    experiment.setup_experiment_from_cfg(cfg)
    result = experiment.train(trainer_cfg=cfg["trainer"])
    print("Result:", result)
