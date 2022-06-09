import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from data.collate_fn import build_collate_fn
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from torchinfo import summary as print_model_summary
from utils.configs import merge_config
from utils.experiment import (
    apply_transforms,
    build_dataset,
    build_initial_transform,
    build_transforms,
    find_lighting_module,
)
from utils.experiment import initialize_environment as _initialize_environment
from utils.experiment import print_to_end
from utils.logging import create_logger
from utils.visualization.utils import plot_samples_from_dataset


class Experiment:
    def __init__(self, cfg=None, const_cfg=None, debug_cfg=None):
        self.const_cfg = cfg["const"] if "const" in cfg else None
        self.cfg_debug = cfg["debug"] if "debug" in cfg else None

    def get_directory(self):
        if not hasattr(self, "experiment_name"):
            raise ValueError(
                "Experiment is not yet initialized. Please call `setup_experiment_from_cfg`."
            )
        return os.path.join(os.getcwd(), self.exp_dir)

    def initialize_environment(self, cfg):
        self.experiment_name = _initialize_environment(cfg)
        self.exp_dir = f"results/{self.experiment_name}"
        # set `experiment_name` as os.environ
        os.environ["EXPERIMENT_NAME"] = self.experiment_name

    def setup_dataset(self, train_dataset, val_dataset, cfg, dataloader=True):
        self._setup_dataset(
            initial_dataset={
                "trn": train_dataset,
                "val": val_dataset,
            },
            dataset_cfg=cfg["dataset"],
            transform_cfg=cfg["transform"],
        )

        if dataloader:
            val_batch_size = (
                cfg["validation"]["batch_size"]
                if ("validation" in cfg and "batch_size" in cfg["validation"])
                else cfg["training"]["batch_size"]
            )
            self._setup_dataloader(
                self.trn_dataset,
                self.val_dataset,
                cfg["dataloader"],
                trn_batch_size=cfg["training"]["batch_size"],
                val_batch_size=val_batch_size,
            )

    def setup_experiment_from_cfg(
        self,
        cfg,
        setup_env=True,
        setup_dataset=True,
        setup_dataloader=True,
        setup_model=True,
        setup_callbacks=True,
    ):
        self.const_cfg = cfg["const"] if "const" in cfg else None
        self.cfg_debug = cfg["debug"] if "debug" in cfg else None

        if setup_env:
            self.initialize_environment(cfg)

        if setup_dataset:
            self._setup_dataset(
                dataset_cfg=cfg["dataset"],
                transform_cfg=cfg["transform"],
            )

        if setup_dataloader:
            val_batch_size = (
                cfg["validation"]["batch_size"]
                if ("validation" in cfg and "batch_size" in cfg["validation"])
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

    def _setup_dataset(
        self, initial_dataset=None, dataset_cfg=None, transform_cfg=None
    ):
        # load data
        print_to_end("-")
        print("[*] Start loading dataset")
        # 1. build initial dataset to read data.
        if initial_dataset is None:
            datasets = build_dataset(dataset_cfg)
        else:
            datasets = initial_dataset
        # datasets: dict{subset_key: torch.utils.data.Dataset, ...}

        # 2. build initial transformation to convert raw data into dictionary.
        if "initial_transform" in dataset_cfg:
            initial_transform = build_initial_transform(
                initial_transform_cfg=dataset_cfg["initial_transform"],
                const_cfg=self.const_cfg,
            )
        else:
            initial_transform = None
        # 3. build list of transformations using `transform` defined in config.
        transforms = build_transforms(
            transform_cfg=transform_cfg,
            const_cfg=self.const_cfg,
            subset_keys=datasets.keys(),
        )
        # 4. actually apply transformations.
        subsets = datasets.keys()
        datasets = {
            subset: apply_transforms(
                datasets[subset], initial_transform, transforms[subset]
            )
            for subset in subsets
        }
        # for now, we mainly consider trn and val subsets.
        trn_dataset, val_dataset = datasets["trn"], datasets["val"]
        # plot samples after data augmentation
        if self.cfg_debug and "view_train_augmentation" in self.cfg_debug:
            save_to = (
                f"{self.exp_dir}/{self.cfg_debug['view_train_augmentation']['save_to']}"
            )
            print(f"[*] Visualizing training samples under `{save_to}")

            plot_samples_from_dataset(
                trn_dataset,
                task=self.const_cfg["task"],
                image_tensor_to_numpy=True,
                unnormalize=True,
                normalization_mean=self.const_cfg["normalization_mean"],
                normalization_std=self.const_cfg["normalization_std"],
                root_dir=self.exp_dir,
                label_map=self.const_cfg["label_map"]
                if "label_map" in self.const_cfg
                else None,
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
        # build configs.
        print("[*] Creating PyTorch `DataLoader`.")
        trn_dataloader_cfg = merge_config(
            dataloader_cfg["base_dataloader"], dataloader_cfg["trn"]
        )
        val_dataloader_cfg = merge_config(
            dataloader_cfg["base_dataloader"], dataloader_cfg["val"]
        )
        if not val_batch_size:
            val_batch_size = trn_batch_size

        # collate_fn
        if "collate_fn" in trn_dataloader_cfg:
            trn_dataloader_cfg["collate_fn"] = build_collate_fn(
                name=trn_dataloader_cfg["collate_fn"]["name"],
                kwargs=trn_dataloader_cfg["collate_fn"]["args"]
            )
        if "collate_fn" in val_dataloader_cfg:
            val_dataloader_cfg["collate_fn"] = build_collate_fn(
                name=val_dataloader_cfg["collate_fn"]["name"],
                kwargs=val_dataloader_cfg["collate_fn"]["args"]
            )

        # dataloader - train
        trn_dataloader = DataLoader(
            trn_dataset,
            batch_size=trn_batch_size,
            **trn_dataloader_cfg,
        )
        # dataloader - val
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            **val_dataloader_cfg,
        )

        self.trn_dataloader, self.val_dataloader = trn_dataloader, val_dataloader

    def _setup_callbacks(
        self, experiment_name=None, wandb_cfg=None, tensorboard_cfg=None
    ):
        if tensorboard_cfg is not None:
            raise NotImplementedError()
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
        lightning_module = find_lighting_module(training_cfg["ID"])
        model = lightning_module(model_cfg, training_cfg)

        if self.cfg_debug and "network_summary" in self.cfg_debug:
            batch_size = 16  # any num:)
            print(f"[*] Model backbone summary(when bs={batch_size}):")

            input_shape = [batch_size] + self.cfg_debug["network_summary"][
                "input_shape"
            ]

            print_model_summary(model, input_size=input_shape)
        # load model from path if specified.
        if "state_dict_path" in model_cfg:
            model.load_state_dict(torch.load(model_cfg["state_dict_path"]))

        self.model = model

    def train(
        self,
        use_existing_trainer=False,
        trainer_cfg={},
        epochs=None,
        root_dir=None,
        state_dict_path="checkpoints/model_state_dict.pth",
        test_after=True,
    ):
        # define pl.Trainer
        if not root_dir:
            root_dir = f"{self.exp_dir}/checkpoints"
        root_dir = os.path.join(root_dir, self.experiment_name)
        if epochs is None:
            epochs = self.model.training_cfg["epochs"]
        if use_existing_trainer:
            train_trainer = self.trainers["train"]
        else:
            train_trainer = pl.Trainer(
                max_epochs=epochs,
                default_root_dir=root_dir,
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

        # train-test using trainer
        train_trainer.fit(self.model, self.trn_dataloader, self.val_dataloader)
        if test_after:
            res = train_trainer.test(self.model, self.val_dataloader)
        else:
            res = {}

        # finish experiment
        if (
            hasattr(self, "logger_and_callbacks")
            and "logger" in self.logger_and_callbacks
        ):
            self.logger_and_callbacks["logger"].experiment.finish()
        if state_dict_path is not None:
            root_path = os.path.dirname(f"{self.exp_dir}/{state_dict_path}")
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            torch.save(self.model.state_dict(), f"{self.exp_dir}/{state_dict_path}")
        return res

    def predict(
        self,
        x,
        use_existing_trainer=False,
        trainer_cfg={},
        root_dir=None,
    ):
        if use_existing_trainer:
            pred_trainer = self.trainers["pred"]
            raise NotImplementedError()
        else:
            if not root_dir:
                root_dir = f"{self.exp_dir}/checkpoints"
            pred_trainer = pl.Trainer(
                default_root_dir=root_dir,
                **trainer_cfg,
            )
        # keep track of trainer
        if not hasattr(self, "trainers"):
            self.trainers = {}
        self.trainers["pred"] = pred_trainer

        pred = pred_trainer.predict(model=self.model, dataloaders=x)
        return pred
