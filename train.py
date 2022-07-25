import os
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
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
    parser.add_argument("--save_path", type=str, default=None)

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
    experiment.initialize_environment(cfg=cfg)
    datasets = experiment.setup_dataset(
        dataset_cfg=cfg["dataset"],
        transform_cfg=cfg["transform"],
    )
    dataloaders = experiment.setup_dataloader(
        datasets=datasets,
        dataloader_cfg=cfg["dataloader"],
    )
    train_dataloader, val_dataloader = dataloaders["trn"], dataloaders["val"]
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

    pl_trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
    )

    # save weights
    if args.save_path is None:
        path_root = os.path.dirname(f"{experiment.exp_dir}/{save_path}")
        filename = f"{experiment.experiment_name}.pth"
    else:
        suffix = Path(args.save_path).suffix
        if suffix == "":
            save_path_root = args.save_path
            filename = f"{experiment.experiment_name}.pth"
        else:
            path_root = os.path.dirname(args.save_path)
            filename = Path(args.save_path).name
    if not os.path.exists(path_root):
        os.makedirs(path_root)
    torch.save(model.state_dict(), f"{path_root}/{filename}")
    # test
    res = pl_trainer.test(model, val_dataloader)

    # log results
    logger_and_callbacks["logger"].experiment.finish()

    print("Result:", res)
    print("Experiment and log dir:", experiment.get_directory())
