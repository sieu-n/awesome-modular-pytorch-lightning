import os
import random

import catalog
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import summary as print_model_summary
from utils.configs import merge_config
from utils.experiment import apply_dataset_mapping as _apply_dataset_mapping
from utils.experiment import apply_transforms
from utils.experiment import build_initial_transform as _build_initial_transform
from utils.experiment import build_transforms as _build_transforms
from utils.experiment import initialize_environment as _initialize_environment
from utils.experiment import print_to_end
from utils.logging import create_logger
from utils.visualization import plot_samples_from_dataset


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
        print_to_end("=")
        self.experiment_name, self.exp_dir = _initialize_environment(cfg)
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

    def setup_dataset(
        self, dataset_cfg, transform_cfg=None, const_cfg=None, subset_to_get=None
    ):
        print_to_end("=")
        print_to_end("=")
        print("[*] Building dataset")
        print_to_end("=")
        print_to_end("=")
        if const_cfg is None:
            const_cfg = self.const_cfg

        # 1. build initial dataset that read data.
        print(
            "(1/5) Building initial dataset from `data.dataset_base_cfg` and `data.dataset_subset_cfg`."
        )
        datasets = self.get_base_dataset(
            dataset_cfg=dataset_cfg,
            subset=None,
        )

        # 2. build initial transformation to convert raw data into dictionary.
        if "initial_transform" in dataset_cfg:
            print(
                "(2/5) Building initial transformations based on `data.initial_transform`."
            )
            initial_transform_cfg = dataset_cfg["initial_transform"]
            initial_transform = self.get_initial_transform(
                initial_transform_cfg=initial_transform_cfg,
                const_cfg=const_cfg,
            )
        else:
            initial_transform = None
            print(
                "(2/5) `data.initial_transform` is not defined in config. Skipping initial transformations."
            )
        # 3. build transformations such as normalization and data augmentation.
        if transform_cfg is not None:
            print("(3/5) Building transformations based on `transform`.")
            transforms = self.get_transform(
                transform_cfg,
                subset=None,  # generate all subsets by default.
                const_cfg=const_cfg,
            )
        else:
            print(
                "(3/5) `transform` is not defined in config. Skipping transformations."
            )
            transforms = {}

        # 4. actually apply transformations.
        subsets = datasets.keys()
        datasets = {
            subset: apply_transforms(
                datasets[subset], initial_transform, transforms.get(subset, None)
            )
            for subset in subsets
        }

        # 5. apply dataset mapping such as "SubsetDataset". These are differnt from transforms
        # since they refernece the entire dataset and can change the length of the dataset as
        # opposed to transforms that are applied to single samples.
        datasets = self.apply_dataset_mapping(
            datasets, mapping_cfg=dataset_cfg.get("mapping", {}), subset_list=subsets
        )

        # plot samples after data augmentation
        if self.debug_cfg and "view_train_augmentation" in self.debug_cfg:
            if "trn" in datasets:
                save_to = f"{self.exp_dir}/{self.debug_cfg['view_train_augmentation']['save_to']}"
                print(f"[*] Visualizing training samples under `{save_to}")

                plot_samples_from_dataset(
                    datasets["trn"],
                    task=self.const_cfg["task"],
                    unnormalize=True,
                    root_dir=self.exp_dir,
                    label_map=self.const_cfg["label_map"]
                    if "label_map" in self.const_cfg
                    else None,
                    **self.debug_cfg["view_train_augmentation"],
                )
            else:
                print(
                    "the 'trn' subset is not defined in the dataset, so `view_train_augmentation` is disabled."
                )

        if subset_to_get is None:
            return datasets
        else:
            return datasets[subset_to_get]

    def setup_callbacks(
        self,
        cfg=None,
        callback_cfg={},
        wandb_cfg=None,
        tensorboard_cfg=None,
    ):
        print_to_end("=")
        if cfg is not None:
            if "wandb" in cfg:
                assert wandb_cfg is None
                wandb_cfg = cfg["wandb"]
            if "tensorboard" in cfg:
                assert tensorboard_cfg is None
                tensorboard_cfg = cfg["tensorboard"]
            if "callbacks" in cfg:
                assert callback_cfg == {}
                callback_cfg = cfg["callbacks"]

        logger_and_callbacks = self._setup_callbacks(
            callback_cfg=callback_cfg,
            experiment_name=self.experiment_name,
            wandb_cfg=wandb_cfg,
            tensorboard_cfg=tensorboard_cfg,
        )
        if "logger" in logger_and_callbacks:
            logger_and_callbacks["logger"].log_hyperparams(cfg)
        return logger_and_callbacks

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
        # build initial dataset to read data.
        if subset is None:
            subset_types = list(dataset_cfg["dataset_subset_cfg"].keys())
        else:
            subset_types = [subset]

        datasets = {}
        for idx, subset_key in enumerate(subset_types):
            print(f"Loading `{subset_key}` dataset subset.")

            dataset_subset_cfg = dataset_cfg["dataset_subset_cfg"][subset_key]
            if dataset_subset_cfg is None:
                dataset_subset_cfg = {}
            assert isinstance(
                dataset_subset_cfg, dict
            ), "Expected a dict, got {}.".format(dataset_subset_cfg)
            dataset_subset_cfg = merge_config(
                dataset_cfg["dataset_base_cfg"], dataset_subset_cfg
            )
            # create dataset.
            datasets[subset_key] = catalog.dataset.build(
                name=dataset_subset_cfg["name"],
                args=dataset_subset_cfg.get("args", {}),
            )

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

    def apply_dataset_mapping(
        self,
        base_datasets,
        mapping_cfg,
        subset=None,
        subset_list=None,
        const_cfg={},
    ):
        if subset_list:
            # constrain subsets when `subset_list` is specified.
            base_datasets = {subset: base_datasets[subset] for subset in subset_list}
        mapped_dataset = _apply_dataset_mapping(
            base_datasets,
            mapping_cfg=mapping_cfg,
            const_cfg=const_cfg,
        )
        # return every subset as a dictionary if `subset` is None
        if subset is None:
            # datasets: dict{subset_key: torch.utils.data.Dataset, ...}
            return mapped_dataset
        else:
            return mapped_dataset[subset]

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
        print_to_end("=")
        print("[*] Creating PyTorch `DataLoader`")
        dataloaders = {}

        base_dataloader_cfg = dataloader_cfg["base_dataloader"]
        if subset_to_get is None:
            it = datasets.items()
        else:
            assert isinstance(datasets, torch.utils.data.Dataset)
            it = [(subset_to_get, datasets)]
        # build dataloader for each subset and apply to the dataset object.
        for subset, dataset in it:
            # build configs.
            print(f"Creating `DataLoader` for subset `{subset}`.")
            subset_dataloader_cfg = merge_config(
                base_dataloader_cfg, dataloader_cfg.get(subset, {})
            )
            # build collate_fn
            if "collate_fn" in subset_dataloader_cfg:
                subset_dataloader_cfg["collate_fn"] = catalog.collate_fn.build(
                    name=subset_dataloader_cfg["collate_fn"]["name"],
                    args=subset_dataloader_cfg["collate_fn"].get("args", {}),
                )
            # build dataloader
            dataloaders[subset] = DataLoader(
                dataset,
                **subset_dataloader_cfg,
            )
        if subset_to_get is None:
            return dataloaders
        else:
            return dataloaders[subset_to_get]

    def _setup_callbacks(
        self,
        experiment_name=None,
        callback_cfg={},
        wandb_cfg=None,
        tensorboard_cfg=None,
    ):
        if tensorboard_cfg is not None:
            raise NotImplementedError()
        if not experiment_name:
            assert hasattr(self, "experiment_name")
            experiment_name = self.experiment_name
        # logger
        logger = create_logger(
            experiment_name=self.experiment_name,
            wandb_cfg=wandb_cfg,
            tensorboard_cfg=tensorboard_cfg,
        )
        # callbacks
        callbacks = []
        for name, cfg in callback_cfg.items():
            callbacks.append(catalog.callbacks.build(**cfg))
        return {
            "logger": logger,
            "callbacks": callbacks,
        }

    def setup_model(self, model_cfg, training_cfg):
        model = catalog.lightning.build_from_cfg(
            model_cfg, training_cfg, self.const_cfg
        )

        print("Model:")
        print(model)

        if self.debug_cfg and "network_summary" in self.debug_cfg:
            batch_size = 16  # any num:)
            print(f"[*] Model backbone summary(when bs={batch_size}):")

            input_shape = [batch_size] + self.debug_cfg["network_summary"][
                "input_shape"
            ]

            print_model_summary(model, input_size=input_shape)
        return model
