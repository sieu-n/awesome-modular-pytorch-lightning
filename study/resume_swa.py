from argparse import ArgumentParser

from main import Experiment
from pytorch_lightning.callbacks import StochasticWeightAveraging
from utils.configs import read_configs

if __name__ == "__main__":
    # read config yaml paths
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--annealing_epochs", type=int, default=0)
    parser.add_argument(
        "--annealing_strategy", type=str, default="cos", choices=["cos", "linear"]
    )
    parser.add_argument("--avg_mode", type=str, default="const", choices=["const"])

    args = parser.parse_args()
    cfg = read_configs(args.configs)

    # write configs
    print("[*] Editing config file.")
    cfg["model"]["state_dict_path"] = args.weights
    if cfg["training"].pop("lr_warmup", None):
        print("warmup removed from config file.")
    cfg["training"]["epochs"] = args.epochs
    print("training.epochs overriden in config file.")
    if args.lr is None:
        args.lr = cfg["training"]["lr"] * 0.3
        cfg["training"]["lr"] = args.lr
        print(f"initializing swa learning rate to {args.lr}")
    if args.avg_mode == "const":
        avg_fn = None
    else:
        raise ValueError("Invalid avg_mode value")
    # build swa callback
    swa_callback = StochasticWeightAveraging(
        swa_lrs=args.lr,
        swa_epoch_start=0.0,
        annealing_epochs=args.annealing_epochs,
        annealing_strategy=args.annealing_strategy,
        avg_fn=avg_fn,
    )
    callbacks = cfg.get("callbacks", [])
    callbacks.append(swa_callback)
    cfg["callbacks"] = callbacks
    # train
    experiment = Experiment(cfg)
    experiment.setup_experiment_from_cfg(cfg)
    result = experiment.train(
        trainer_cfg=cfg["trainer"],
        epochs=cfg["training"]["epochs"],
    )
    print("Result:", result)
    print("Experiment and log dir:", experiment.get_directory())
