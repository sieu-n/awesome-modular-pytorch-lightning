from argparse import ArgumentParser

import pytorch_lightning as pl

from utils.configs import read_configs
from utils.experiment import setup_env, build_dataset

if __name__ == "__main__":
    # read config yaml paths
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", required=True)

    args = parser.parse_args()
    config = read_configs(args.configs)

    setup_env(config)

    # load data
    datasets = build_dataset()
    trn_dataset, val_dataset = datasets["trn"], datasets["val"]

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
