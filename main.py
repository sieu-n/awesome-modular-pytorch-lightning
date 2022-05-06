from argparse import ArgumentParser

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from utils.configs import read_configs, merge_config
from utils.experiment import setup_env, build_dataset, print_to_end
from utils.visualization.vision import PlotSamples


if __name__ == "__main__":
    # read config yaml paths
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", required=True)

    args = parser.parse_args()
    cfg = read_configs(args.configs)

    setup_env(cfg)

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

    # dataloader
    print("[*] Creating PyTorch `DataLoader`.")
    trn_dataloader_cfg = merge_config(cfg["dataloader"]["base_dataloader"], cfg["dataloader"]["trn"])
    val_dataloader_cfg = merge_config(cfg["dataloader"]["base_dataloader"], cfg["dataloader"]["val"])
    trn_dataloader = DataLoader(
        trn_dataset,
        batch_size=cfg["training"]["batch_size"],
        **trn_dataloader_cfg,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg["validation"]["batch_size"],
        **val_dataloader_cfg,
    )

    # callbacks
    print_to_end("-")
    wandb_callback = 0#TODO
    checkpoint_callback = 0#TODO
    ...

    # model
    model = 0# TODO(instance of pl.LightningModule)
    trainer = pl.Trainer()


    trainer.fit(model, trn_dataloader, val_dataloader)
