import os
import pandas as pd
from argparse import ArgumentParser
import pickle
import torch
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
    
    parser.add_argument("--root_dir", type=str, default="/home/hackathon/jupyter/storage/CT/ct_new")
    parser.add_argument("--to", type=str, default="submission.csv")

    args = parser.parse_args()
    cfg = read_configs(args.configs)
    cfg["wandb"]["offline"] = True

    cfg["model"]["state_dict_path"] = args.weights
    cfg["model"]["is_ckpt"] = args.is_ckpt
    cfg["dataset"]["dataset_subset_cfg"]["test"]["args"]["root"] = args.root_dir
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
    epochs = cfg["training"]["epochs"]

    pl_trainer = pl.Trainer(
        max_epochs=epochs,
        **logger_and_callbacks,
        **cfg["trainer"],
    )

    res = pl_trainer.predict(model, pred_dataloader)
    pred_argmax = torch.cat(res).argmax(1)

    print(f"saving to: {args.to}")

    pd.DataFrame(dict(
        filename=sorted(os.listdir(args.root_dir)),
        result=pred_argmax,
    )).to_csv(args.to, index=False)
