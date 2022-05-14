import os

import lightning.trainers as trainers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from utils.configs import merge_config
from utils.experiment import (
    build_dataset,
    build_network,
    initialize_environment,
    print_to_end,
)
from utils.logging import create_logger
from utils.visualization.utils import plot_samples_from_dataset


class Experiment:
    def __init__(self, cfg={}):
        self.cfg_const = cfg["const"] if "const" in cfg else None
        self.cfg_debug = cfg["debug"] if "debug" in cfg else None

    def setup_experiment_from_cfg(
        self,
        cfg,
        setup_env=True,
        setup_dataset=True,
        setup_dataloader=True,
        setup_model=True,
        setup_callbacks=True,
    ):
        self.cfg_const = cfg["const"] if "const" in cfg else None
        self.cfg_debug = cfg["debug"] if "debug" in cfg else None

        if setup_env:
            self.experiment_name = initialize_environment(cfg)
        self.exp_dir = f"results/{self.experiment_name}"
        # set `experiment_name` as os.environ
        os.environ["EXPERIMENT_NAME"] = self.experiment_name
        if setup_dataset:
            self._setup_dataset(cfg["dataset"], cfg["transform"])

        if setup_dataloader:
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
        if setup_callbacks:
            self._setup_callbacks(
                experiment_name=self.experiment_name,
                wandb_cfg=cfg.get("wandb", None),
                tensorboard_cfg=cfg.get("tensorboard", None),
            )
        if setup_model:
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
            save_to = (
                f"{self.exp_dir}/{self.cfg_debug['view_train_augmentation']['save_to']}"
            )
            print(f"[*] Visualizing training samples under `{save_to}")

            plot_samples_from_dataset(
                trn_dataset,
                task=self.cfg_const["task"],
                image_tensor_to_numpy=True,
                unnormalize=True,
                normalization_mean=self.cfg_const["normalization_mean"],
                normalization_std=self.cfg_const["normalization_std"],
                root_dir=self.exp_dir,
                **self.cfg_debug["view_train_augmentation"],
            )

        self.trn_dataset, self.val_dataset = trn_dataset, val_dataset

    def _setup_dataloader(
        self,
        trn_dataset,
        val_dataset,
        dataloader_cfg,
        trn_batch_size,
        val_batch_size=None,
    ):
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

    def _setup_callbacks(
        self, experiment_name=None, wandb_cfg=None, tensorboard_cfg=None
    ):
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

        self.logger_and_callbacks = {
            "logger": logger,
            "callbacks": [checkpoint_callback, lr_callback],
        }

    def _setup_model(self, model_cfg, training_cfg):
        # model
        net = build_network(model_cfg)
        model = getattr(trainers, training_cfg["ID"])(training_cfg, net)

        self.network, self.model = net, model

    def train(
        self,
        use_existing_trainer=False,
        trainer_cfg={},
        epochs=None,
        root_dir=None,
        test_after=True,
    ):
        if use_existing_trainer:
            train_trainer = self.trainers["train"]
            raise NotImplementedError()
        else:
            if not root_dir:
                root_dir = f"{self.exp_dir}/checkpoints"
            train_trainer = pl.Trainer(
                max_epochs=epochs,
                default_root_dir=f"{self.exp_dir}/checkpoints",
                **(
                    self.logger_and_callbacks
                    if hasattr(self, "logger_and_callbacks")
                    else {}
                ),
                **trainer_cfg,
            )
        # keep track of trainer
        if not hasattr(self, "trainers"):
            self.trainers = {}
        self.trainers["train"] = train_trainer

        train_trainer.fit(self.model, self.trn_dataloader, self.val_dataloader)
        if test_after:
            res = train_trainer.test(self.model, self.val_dataloader)
        else:
            res = {}
        if (
            hasattr(self, "logger_and_callbacks")
            and "logger" in self.logger_and_callbacks
        ):
            self.logger_and_callbacks["logger"].experiment.finish()
        return res
