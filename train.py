from argparse import ArgumentParser

from main import Experiment
from utils.configs import read_configs

if __name__ == "__main__":
    # read config yaml paths
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", required=True)
    parser.add_argument("--root_dir", type=str, default=None)

    args = parser.parse_args()
    cfg = read_configs(args.configs)

    # train
    experiment = Experiment(cfg)
    experiment.setup_experiment_from_cfg(cfg)
    result = experiment.train(
        trainer_cfg=cfg["trainer"],
        root_dir=args.root_dir,
        epochs=cfg["training"]["epochs"],
    )
    print("Result:", result)
    print("Experiment and log dir:", experiment.get_directory())
