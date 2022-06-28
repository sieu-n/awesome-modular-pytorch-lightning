import os
import random
import re
import lightning
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchinfo import summary as print_model_summary

from data.collate_fn import build_collate_fn
from utils.callbacks import build_callback
from utils.configs import merge_config
from utils.experiment import (
    apply_transforms,
    build_dataset as _build_dataset,
    build_initial_transform as _build_initial_transform,
    build_transforms as _build_transforms,
)
from utils.experiment import initialize_environment as _initialize_environment
from utils.experiment import print_to_end
from utils.logging import create_logger
from utils.pretrained import load_model_weights
from utils.visualization.utils import plot_samples_from_dataset


class Experiment:
    def __init__(self, cfg=None, const_cfg=None, debug_cfg=None):
        if cfg is None:
            self.const_cfg = const_cfg
            self.debug_cfg = debug_cfg
        else:
            self.const_cfg = cfg["const"] if "const" in cfg else None
            self.debug_cfg = cfg["debug"] if "debug" in cfg else None

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
        # initialize seed
        if "seed" in cfg:
            t = type(cfg["seed"])
            if t == bool:
                self.set_seed()
            elif t == int:
                self.set_seed(cfg["seed"])
            else:
                self.set_seed(int(cfg["seed"]))
        else:
            print("For your information, random seed is not set")

    def set_seed(self, seed=3407):
        """
        Set seed for reproducibility. By default, set seed to 3407 according to the following paper😅 :
        Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for
        computer vision, 2021
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def setup_dataset(self, dataset_cfg, transform_cfg, const_cfg=None, subset_to_get=None):
        if const_cfg is None:
            const_cfg = self.const_cfg

        # 1. build initial dataset that read data.
        datasets = self.get_base_dataset(
            dataset_cfg=dataset_cfg,
            subset=None,
        )

        # 2. build initial transformation to convert raw data into dictionary.
        if "initial_transform" in dataset_cfg:
            initial_transform_cfg = dataset_cfg["initial_transform"]
            initial_transform = self.get_initial_transform(
                initial_transform_cfg=initial_transform_cfg,
                const_cfg=const_cfg,
            )
        else:
            initial_transform = None

        # 3. build transformations such as normalization and data augmentation.
        transforms = self.get_transform(
            transform_cfg,
            subset=None,    # generate all subsets by default.
            const_cfg=const_cfg,
        )

        # 4. actually apply transformations.
        subsets = datasets.keys()
        datasets = {
            subset: apply_transforms(
                datasets[subset], initial_transform, transforms[subset]
            )
            for subset in subsets
        }

        # plot samples after data augmentation
        if self.debug_cfg and "view_train_augmentation" in self.debug_cfg:
            if "trn" in datasets:
                save_to = (
                    f"{self.exp_dir}/{self.debug_cfg['view_train_augmentation']['save_to']}"
                )
                print(f"[*] Visualizing training samples under `{save_to}")

                plot_samples_from_dataset(
                    datasets["trn"],
                    task=self.const_cfg["task"],
                    image_tensor_to_numpy=True,
                    unnormalize=True,
                    normalization_mean=self.const_cfg["normalization_mean"],
                    normalization_std=self.const_cfg["normalization_std"],
                    root_dir=self.exp_dir,
                    label_map=self.const_cfg["label_map"]
                    if "label_map" in self.const_cfg
                    else None,
                    **self.debug_cfg["view_train_augmentation"],
                )
            else:
                print("the 'trn' subset is not defined in the dataset, so `view_train_augmentation` is disabled.")

        if subset_to_get is None:
            return datasets
        else:
            return datasets[subset_to_get]

    def setup_experiment_from_cfg(
        self,
        cfg,
        setup_model=True,
        setup_callbacks=True,
    ):
        if setup_callbacks:
            self._setup_callbacks(
                callback_cfg_list=cfg.get("callbacks", []),
                experiment_name=self.experiment_name,
                wandb_cfg=cfg.get("wandb", None),
                tensorboard_cfg=cfg.get("tensorboard", None),
            )
            if "logger" in self.logger_and_callbacks:
                self.logger_and_callbacks["logger"].log_hyperparams(cfg)

        if setup_model:
            self._setup_model(cfg["model"], cfg["training"])

    def get_base_dataset(
        self,
        dataset_cfg,
        subset=None,
    ):
        """
        Build torch.utils.data.Dataset objects based on cfg["dataset"]
        Parameters
        ----------
        dataset_cfg: dict
            The value of cfg["dataset"] is expected to be passed to the constructor.
        base_dataset: torch.utils.data.Dataset, default=None
        subset: str, default=None
        Returns
        -------
        dict[str: torch.utils.data.Dataset], torch.utils.data.Dataset:
            Returns dictionary containing the torch.utils.data.Dataset objects for each subset. If a value for `subset`
            is provided, a single dataset corresponding to the subset value is returned.
        """
        print_to_end("-")
        print("[*] Start loading dataset")
        # build initial dataset to read data.
        datasets = _build_dataset(dataset_cfg)

        # return every subset as a dictionary if `subset` is None
        if subset is None:
            # datasets: dict{subset_key: torch.utils.data.Dataset, ...}
            return datasets
        else:
            return datasets[subset]

    def get_initial_transform(
        self,
        initial_transform_cfg,
        const_cfg={},
    ):
        """
        Build initial transformation to convert raw data into dictionary.
        """
        return _build_initial_transform(
            initial_transform_cfg=initial_transform_cfg,
            const_cfg=const_cfg,
        )

    def get_transform(
        self,
        transform_cfg,
        subset=None,
        const_cfg={},
    ):
        # build list of transformations using arguments of cfg["transform"]
        transforms = _build_transforms(
            transform_cfg=transform_cfg,
            const_cfg=const_cfg,
        )
        # return every subset as a dictionary if `subset` is None
        if subset is None:
            # datasets: dict{subset_key: torch.utils.data.Dataset, ...}
            return transforms
        else:
            return transforms[subset]

    def setup_dataloader(
        self,
        datasets,
        dataloader_cfg,
        subset_to_get=None,
    ):
        dataloaders = {}

        base_dataloader_cfg = dataloader_cfg["base_dataloader"]
        if subset_to_get is None:
            it = datasets.items()
        else:
            assert isinstance(datasets, torch.utils.data.Dataset)
            it = [(subset_to_get, datasets)]
        for subset, dataset in it:
            # build configs.
            print("[*] Creating PyTorch `DataLoader`.")
            dataloader_cfg = merge_config(
                base_dataloader_cfg, dataloader_cfg.get(subset, {})
            )
            # build collate_fn
            if "collate_fn" in dataloader_cfg:
                dataloader_cfg["collate_fn"] = build_collate_fn(
                    name=dataloader_cfg["collate_fn"]["name"],
                    kwargs=dataloader_cfg["collate_fn"]["args"],
                )
            # build dataloader
            dataloaders.append(DataLoader(
                dataset,
                **dataloader_cfg,
            ))
        if subset_to_get is None:
            return dataloaders
        else:
            return dataloaders[subset_to_get]

    def _setup_callbacks(
        self,
        experiment_name=None,
        callback_cfg_list=[],
        wandb_cfg=None,
        tensorboard_cfg=None,
    ):
        if tensorboard_cfg is not None:
            raise NotImplementedError()
        if not experiment_name:
            assert hasattr(self, "experiment_name")
            experiment_name = self.experiment_name
        # logger
        print_to_end("-")
        logger = create_logger(
            experiment_name=self.experiment_name,
            wandb_cfg=wandb_cfg,
            tensorboard_cfg=tensorboard_cfg
        )
        # callbacks
        callbacks = []
        for callback_cfg in callback_cfg_list:
            callbacks.append(build_callback(callback_cfg))
        self.logger_and_callbacks = {
            "logger": logger,
            "callbacks": callbacks,
        }

    def _setup_model(self, model_cfg, training_cfg):
        # model
        lightning_module = lightning.get(training_cfg["ID"])
        model = lightning_module(model_cfg, training_cfg, self.const_cfg)

        if self.debug_cfg and "network_summary" in self.debug_cfg:
            batch_size = 16  # any num:)
            print(f"[*] Model backbone summary(when bs={batch_size}):")

            input_shape = [batch_size] + self.debug_cfg["network_summary"][
                "input_shape"
            ]

            print_model_summary(model, input_size=input_shape)
        # load model from path if specified.
        if "state_dict_path" in model_cfg:
            load_model_weights(
                model=model,
                state_dict_path=model_cfg["state_dict_path"],
                is_ckpt=model_cfg.get("is_ckpt", False),
            )

        self.model = model

    def train(
        self,
        use_existing_trainer=False,
        trainer_cfg={},
        epochs=None,
        root_dir=None,
        save_path="checkpoints/model_state_dict.pth",
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
            if "precision" in trainer_cfg:
                assert (
                    trainer_cfg["precision"] == 32 or self.model.automatic_optimization
                ), "Manual optimization \
                        using amp is not yet supported"
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
            res = self.test(trainer_name="train")
        else:
            res = {}

        # finish experiment
        if (
            hasattr(self, "logger_and_callbacks")
            and "logger" in self.logger_and_callbacks
        ):
            self.logger_and_callbacks["logger"].experiment.finish()
        if save_path is not None:
            root_path = os.path.dirname(f"{self.exp_dir}/{save_path}")
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            torch.save(self.model.state_dict(), f"{self.exp_dir}/{save_path}")
        return res

    def test(
        self,
        test_dataloader=None,
        trainer_name=None,
        trainer_cfg=None,
        root_dir=None,
    ):
        if not test_dataloader:
            test_dataloader = self.val_dataloader
        if trainer_name:
            test_trainer = self.trainers[trainer_name]
        elif trainer_cfg:
            if not root_dir:
                root_dir = f"{self.exp_dir}/checkpoints"
            test_trainer = pl.Trainer(
                default_root_dir=root_dir,
                **trainer_cfg,
            )
        else:
            assert (
                "test" in self.trainers
            ), "Please specify the trainer to use for testing such as batch size and root dir."
            test_trainer = self.trainers["test"]
        # keep track of trainer
        if not hasattr(self, "trainers"):
            self.trainers = {}
        self.trainers["test"] = test_trainer

        res = test_trainer.test(self.model, test_dataloader)
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
