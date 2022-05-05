from argparse import ArgumentParser

import pytorch_lightning as pl

from utils.configs import read_configs
from utils.experiment import setup_env, build_dataset
from utils.visualization.vision import PlotSamples


if __name__ == "__main__":
    # read config yaml paths
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", required=True)

    args = parser.parse_args()
    cfg = read_configs(args.configs)

    setup_env(cfg)

    # load data
    datasets = build_dataset(cfg["dataset"], cfg["transform"])
    trn_dataset, val_dataset = datasets["trn"], datasets["val"]

    if "vis" in cfg and "view_train_augmentation" in cfg["vis"]:
        PlotSamples(**cfg["vis"]["view_train_augmentation"])

    trn_dataloader = 0# TODO
    val_dataloader = 0# TODO

    # callbacks
    wandb_callback = 0#TODO
    checkpoint_callback = 0#TODO
    ...

    # model
    model = 0# TODO(instance of pl.LightningModule)
    trainer = pl.Trainer()


    trainer.fit(model, trn_dataloader, val_dataloader)
