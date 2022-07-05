import os
from argparse import ArgumentParser
from copy import deepcopy

import pytorch_lightning as pl
from main import Experiment
from utils.configs import compute_links, merge_config, read_configs
from utils.logging import log_to_wandb


def set_key_to(d, key, value):
    # Simply traversing `d` is indeed more complex because `key` might not exist in d.
    # For example, given key="training.optimizer.lr" and value="0.01",
    # create an empty dictionary: d_to_push = { "training": { "optimizer": { } } }
    d_to_push = {"WRAPPER": None}
    prev_key = "WRAPPER"
    _d_to_push = d_to_push
    for k in key.split("."):
        # 'mapping.[0].[1].[0]' -> ["mapping", "0]", "1]", "0]"]
        dict_key = k.split("[")[0]
        _d_to_push[prev_key] = {dict_key: None}
        _d_to_push = _d_to_push[prev_key]
        prev_key = dict_key
        # loop through ["0]", "1]", "0]"]
        for list_idx in k.split("[")[1:]:
            assert list_idx[-1] == "]", f"{list_idx} was given."
            list_idx = int(list_idx[:-1])
            _d_to_push[prev_key] = [{} for _ in range(list_idx + 1)]
            _d_to_push[prev_key][list_idx] = None
            _d_to_push = _d_to_push[prev_key]
            prev_key = list_idx
    # fill in 0.01: d_to_push = { "training": { "optimizer": { "lr": 0.01 } } }
    _d_to_push[prev_key] = value
    merged = merge_config(cfg_base=d, cfg_from=d_to_push["WRAPPER"])
    return merged


def get_data_size_schedule(args, num_train_samples):
    if args.range:
        init_samples, step = int(args.range[0]), int(args.range[1])
        if len(args.range) == 2:
            stop = num_train_samples
        elif len(args.range) == 3:
            stop = int(args.range[2])
        else:
            raise ValueError(f"Invalid range value: {args.range}")
        data_size_schedule = list(range(init_samples, stop + 1, step))
    elif args.range_percent:
        init_samples = int(float(args.range_percent[0]) * num_train_samples)
        step = int(float(args.range_percent[1]) * num_train_samples)
        data_size_schedule = list(range(init_samples, num_train_samples + 1, step))
    elif args.size_at_cycle:
        data_size_schedule = [int(n) for n in args.size_at_cycle]
    elif args.size_at_cycle_percent:
        data_size_schedule = [
            int(float(n) * num_train_samples) for n in args.size_at_cycle_percent
        ]
    return data_size_schedule


if __name__ == "__main__":
    # read config yaml paths
    parser = ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--offline", action="store_true", default=False)
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--set_same_group", default=False, action="store_true")
    parser.add_argument(
        "-v",
        "--value",
        nargs="+",
        help="list of values",
        required=True,
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-t", "--dtype", default="str", type=str, choices=["str", "int", "float"]
    )
    args = parser.parse_args()
    cfg = read_configs(args.configs)

    if args.name is not None:
        cfg["name"] = args.name
    if args.group:
        cfg["wandb"]["group"] = args.group
    if args.set_same_group:
        cfg["wandb"]["group"] = cfg["name"]
    if args.offline:
        cfg["wandb"]["offline"] = True

    # set hparam sweep schedule
    dtype_map = {
        "int": int,
        "float": float,
        "str": str,
    }
    values = list(map(dtype_map[args.dtype], args.value))
    print("Hparam schedule: " + str(values))

    results = []
    for idx, value in enumerate(values):
        print(f"Cycle # {idx} / {len(values)} | {args.key}: {value}")
        cycle_cfg = deepcopy(cfg)
        cycle_cfg["name"] = f"{cycle_cfg['name']}-cycle_{idx}-{args.key}_{value}"

        # set `args.key` in config to `value`
        cycle_cfg = set_key_to(cycle_cfg, args.key, value)
        cycle_cfg = compute_links(cycle_cfg)
        ################################################################
        # build experiment
        ################################################################
        experiment = Experiment(cycle_cfg)
        experiment.initialize_environment(cfg=cycle_cfg)

        datasets = experiment.setup_dataset(
            dataset_cfg=cycle_cfg["dataset"],
            transform_cfg=cycle_cfg["transform"],
        )
        dataloaders = experiment.setup_dataloader(
            datasets=datasets,
            dataloader_cfg=cycle_cfg["dataloader"],
        )
        train_dataloader, val_dataloader = dataloaders["trn"], dataloaders["val"]
        # build model and callbacks
        model = experiment.setup_model(
            model_cfg=cycle_cfg["model"], training_cfg=cycle_cfg["training"]
        )
        logger_and_callbacks = experiment.setup_callbacks(cfg=cycle_cfg)
        ################################################################
        # train
        ################################################################
        save_path = ("checkpoints/model_state_dict.pth",)
        if not args.root_dir:
            root_dir = os.path.join(
                f"{experiment.exp_dir}/checkpoints", experiment.experiment_name
            )
        else:
            root_dir = os.path.join(args.root_dir, experiment.experiment_name)
        epochs = cycle_cfg["training"]["epochs"]
        # lightning trainer
        pl_trainer = pl.Trainer(
            max_epochs=epochs,
            default_root_dir=root_dir,
            **logger_and_callbacks,
            **cycle_cfg["trainer"],
        )
        # train
        pl_trainer.fit(
            model,
            train_dataloader,
            val_dataloader,
        )
        # test
        res = pl_trainer.test(model, val_dataloader)
        # log results
        logger_and_callbacks["logger"].experiment.finish()

        res[0][args.key] = value
        print("Result:", res)
        results.append(res[0])

    print("Final results:", results)
    if "wandb" in cfg:
        log_to_wandb(
            results,
            exp_name=f"final-results-{cfg['name']}",
            group=cfg["wandb"].get("group", None),
            project=cfg["wandb"].get("project", None),
        )
