import os
import lightning.trainers as trainers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from utils.configs import merge_config, read_configs
from utils.experiment import (
    build_dataset,
    build_network,
    create_logger,
    print_to_end,
    setup_env,
)
from utils.visualization.vision import PlotSamples


class Experiment():
    def __init__(self, cfg={}):
        self.cfg_const = cfg["const"] if "const" in cfg else None
        self.cfg_debug = cfg["debug"] if "debug" in cfg else None

    def setup_experiment_from_cfg(self, cfg):
        self.cfg_const = cfg["const"] if "const" in cfg else None
        self.cfg_debug = cfg["debug"] if "debug" in cfg else None

        self.experiment_name = setup_env(cfg)
        # set `experiment_name` as os.environ
        os.environ["EXPERIMENT_NAME"] = self.experiment_name
        self._setup_dataset(cfg["dataset"], cfg["transform"])

        val_batch_size = (
            cfg["validation"]["batch_size"]
            if "batch_size" in cfg["validation"]
            else cfg["training"]["batch_size"]
        )
        self._setup_dataloader(
            self.trn_dataset,
            self.val_dataset,
            cfg["dataloader"],
            trn_batch_size=cfg["training"]["batch_size"],
            val_batch_size=val_batch_size,
        )
        self._setup_callbacks(experiment_name=self.experiment_name)
        self._setup_model(cfg["model"], cfg["training"])

        if "logger" in self.logger_and_callbacks:
            self.logger_and_callbacks["logger"].log_hyperparams(cfg)

    def _setup_dataset(self, dataset_cfg, transform_cfg):
        # load data
        print_to_end("-")
        print("[*] Start loading dataset")
        datasets = build_dataset(dataset_cfg, transform_cfg)
        trn_dataset, val_dataset = datasets["trn"], datasets["val"]

        # plot samples after data augmentation
        if self.cfg_debug and "view_train_augmentation" in self.cfg_debug:
            print(
                f"[*] Visualizing training samples under `{self.cfg_debug['view_train_augmentation']['save_to']}"
            )

            reverse_normalization = {}
            reverse_normalization["normalization_mean"] = self.cfg_const["normalization_mean"]
            reverse_normalization["normalization_std"] = self.cfg_const["normalization_std"]

            PlotSamples(
                trn_dataset,
                **reverse_normalization,
                **self.cfg_debug["view_train_augmentation"],
            )

        self.trn_dataset, self.val_dataset = trn_dataset, val_dataset

    def _setup_dataloader(self, trn_dataset, val_dataset, dataloader_cfg, trn_batch_size, val_batch_size=None):
        # dataloader - train
        print("[*] Creating PyTorch `DataLoader`.")
        trn_dataloader_cfg = merge_config(
            dataloader_cfg["base_dataloader"], dataloader_cfg["trn"]
        )
        val_dataloader_cfg = merge_config(
            dataloader_cfg["base_dataloader"], dataloader_cfg["val"]
        )
        trn_dataloader = DataLoader(
            trn_dataset,
            batch_size=trn_batch_size,
            **trn_dataloader_cfg,
        )
        # dataloader - val
        if not val_batch_size:
            val_batch_size = trn_batch_size

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            **val_dataloader_cfg,
        )

        self.trn_dataloader, self.val_dataloader = trn_dataloader, val_dataloader

    def _setup_callbacks(self, experiment_name=None, wandb_cfg=None, tensorboard_cfg=None):
        if not experiment_name:
            assert hasattr(self, "experiment_name")
            experiment_name = self.experiment_name
        # callbacks
        print_to_end("-")
        logger_cfg = {
            "wandb_cfg": wandb_cfg,
            "tensorboard_cfg": tensorboard_cfg,
        }
        logger = create_logger(experiment_name=self.experiment_name, **logger_cfg)
        checkpoint_callback = ModelCheckpoint(
            monitor="epoch/val_performance", save_last=True, save_top_k=1, mode="max"
        )
        lr_callback = LearningRateMonitor(logging_interval="epoch")

        self.logger_and_callbacks = {"logger": logger, "callbacks": [checkpoint_callback, lr_callback]}

    def _setup_model(self, model_cfg, training_cfg):
        # model
        net = build_network(model_cfg)
        model = getattr(trainers, training_cfg["ID"])(training_cfg, net)

        self.network, self.model = net, model

    def train(self, use_existing_trainer=False, trainer_cfg={}, epochs=None, root_dir=None, test=True):
        if use_existing_trainer:
            train_trainer = self.trainers["train"]
            raise NotImplementedError()
        else:
            if not root_dir:
                root_dir = f"results/checkpoints/{self.experiment_name}"
            train_trainer = pl.Trainer(
                max_epochs=epochs,
                default_root_dir=f"results/checkpoints/{self.experiment_name}",
                **(self.logger_and_callbacks if hasattr(self, "logger_and_callbacks") else {}),
                **trainer_cfg,
            )
        # keep track of trainer
        if not hasattr(self, "trainers"):
            self.trainers = {}
        self.trainers["train"] = train_trainer

        train_trainer.fit(self.model, self.trn_dataloader, self.val_dataloader)
        if test:
            res = self.trainer.test(self.model, self.val_dataloader)
            return res
