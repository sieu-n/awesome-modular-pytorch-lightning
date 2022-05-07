from argparse import ArgumentParser

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import lightning.trainers as trainers

from utils.configs import read_configs, merge_config
from utils.experiment import (setup_env, build_dataset, print_to_end, build_network, create_logger)
from utils.visualization.vision import PlotSamples


if __name__ == "__main__":
    # read config yaml paths
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", required=True)

    args = parser.parse_args()
    cfg = read_configs(args.configs)

    experiment_name = setup_env(cfg)

    # load data
    print_to_end("-")
    print("[*] Start loading dataset")
    datasets = build_dataset(cfg["dataset"], cfg["transform"])
    trn_dataset, val_dataset = datasets["trn"], datasets["val"]

    # plot samples after data augmentation
    if "debug" in cfg and "view_train_augmentation" in cfg["debug"]:
        print(f"[*] Visualizing training samples under `{cfg['debug']['view_train_augmentation']['save_to']}")

        reverse_normalization = {}
        reverse_normalization["normalization_mean"] = cfg["const"]["normalization_mean"]
        reverse_normalization["normalization_std"] = cfg["const"]["normalization_std"]

        PlotSamples(trn_dataset, **reverse_normalization, **cfg["debug"]["view_train_augmentation"])

    # dataloader - train
    print("[*] Creating PyTorch `DataLoader`.")
    trn_dataloader_cfg = merge_config(cfg["dataloader"]["base_dataloader"], cfg["dataloader"]["trn"])
    val_dataloader_cfg = merge_config(cfg["dataloader"]["base_dataloader"], cfg["dataloader"]["val"])
    trn_dataloader = DataLoader(
        trn_dataset,
        batch_size=cfg["training"]["batch_size"],
        **trn_dataloader_cfg,
    )
    # dataloader - val
    val_batch_size = cfg["validation"]["batch_size"] if "batch_size" in cfg["validation"] \
        else cfg["training"]["batch_size"]
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        **val_dataloader_cfg,
    )

    # callbacks
    print_to_end("-")
    logger = create_logger(cfg, experiment_name=experiment_name)
    checkpoint_callback = ModelCheckpoint(monitor="val_performance",
                                          save_last=True,
                                          save_top_k=1,
                                          mode='max')
    lr_callback = LearningRateMonitor(logging_interval='epoch')

    # model
    net = build_network(cfg["model"])
    model = getattr(trainers, cfg["training"]["ID"])(cfg, net)

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["epochs"],
        default_root_dir=f"results/checkpoints/{experiment_name}",
        callbacks=[checkpoint_callback, lr_callback],
        logger=logger,
        **cfg["trainer"],
    )

    trainer.fit(model, trn_dataloader, val_dataloader)
    res = trainer.test(model, val_dataloader)
    print(res)
