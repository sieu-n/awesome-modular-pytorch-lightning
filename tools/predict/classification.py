import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from main import Experiment
from utils.configs import read_configs

if __name__ == "__main__":
    # read config yaml paths
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", required=True)
    parser.add_argument("-d", "--dataset_key", required=True)
    parser.add_argument("-w", "--weights", required=True)
    parser.add_argument("--is_ckpt", default=False, action="store_true")
    parser.add_argument("--root_dir", type=str, default=None)

    args = parser.parse_args()
    cfg = read_configs(args.configs)
    cfg["wandb"]["offline"] = True

    cfg["model"]["state_dict_path"] = args.weights
    cfg["model"]["is_ckpt"] = args.is_ckpt
    # initialize experiment
    experiment = Experiment(cfg)
    experiment.initialize_environment(cfg=cfg)
    datasets = experiment.setup_dataset(
        dataset_cfg=cfg["dataset"],
        transform_cfg=cfg["transform"],
    )
    dataloaders = experiment.setup_dataloader(
        datasets=datasets,
        dataloader_cfg=cfg["dataloader"],
    )
    pred_dataloader = dataloaders[args.dataset_key]
    model = experiment.setup_model(model_cfg=cfg["model"], training_cfg=cfg["training"])
    logger_and_callbacks = experiment.setup_callbacks(cfg=cfg)

    # train
    save_path = "checkpoints/model_state_dict.pth"
    if not args.root_dir:
        root_dir = os.path.join(
            f"{experiment.exp_dir}/checkpoints", experiment.experiment_name
        )
    else:
        root_dir = os.path.join(args.root_dir, experiment.experiment_name)
    epochs = cfg["training"]["epochs"]

    pl_trainer = pl.Trainer(
        max_epochs=epochs,
        default_root_dir=root_dir,
        **logger_and_callbacks,
        **cfg["trainer"],
    )

    res = pl_trainer.predict(model, pred_dataloader)
    print("Result:", res)
