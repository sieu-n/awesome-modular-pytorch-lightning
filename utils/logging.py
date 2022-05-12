import wandb
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm


def create_logger(
    experiment_name, wandb_cfg=None, tensorboard_cfg=None, entire_cfg=None
):
    assert (
        (wandb_cfg is None) + (tensorboard_cfg is None)
    ) == 1, "Only one config should be specified."
    if wandb_cfg:
        print(f"wandb name: {experiment_name}")
        if "project" not in wandb_cfg:
            wandb_cfg["project"] = "modular-pytorch-lightning-extensions"
        logger = WandbLogger(
            name=experiment_name,
            **wandb_cfg,
        )
    elif tensorboard_cfg:
        raise NotImplementedError()
    else:
        print("[*] No logger is specified, returning `None`.")
        return None
    # log hparams.
    if entire_cfg:
        logger.log_hyperparams(entire_cfg)
    return logger


def log_to_wandb(log, exp_name, group=None, project=None):
    """
    log: list[float] or list[dict]
    """
    wandb.init(
        name=exp_name,
        project=project,
        group=group,
    )

    def recurse_dict(d, key_record):
        if isinstance(d, dict):
            # recurse
            output = {}
            for k in d.keys():
                subdict = recurse_dict(d[k], f"{key_record}/{k}")
                output = {**output, **subdict}
            return output
        else:
            return {key_record[1:]: d}

    # log per-step metrics
    for step_log in tqdm(log):
        if type(step_log) == dict:
            step_log = recurse_dict(step_log, "")
        wandb.log(step_log)
    wandb.finish()
