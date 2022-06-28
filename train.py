from argparse import ArgumentParser

from main import Experiment
from utils.configs import read_configs

if __name__ == "__main__":
    # read config yaml paths
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--offline", action="store_true", default=False)
    parser.add_argument("--root_dir", type=str, default=None)

    args = parser.parse_args()
    cfg = read_configs(args.configs)
    if args.name is not None:
        cfg["name"] = args.name
    if args.group:
        cfg["wandb"]["group"] = args.group
    if args.offline:
        cfg["wandb"]["offline"] = True
    # initialize experiment
    experiment = Experiment(cfg)
    experiment.initialize_environment(cfg)
    experiment.setup_experiment_from_cfg(cfg)
    # train
    result = experiment.train(
        trainer_cfg=cfg["trainer"],
        root_dir=args.root_dir,
        epochs=cfg["training"]["epochs"],
    )
    print("Result:", result)
    print("Experiment and log dir:", experiment.get_directory())
