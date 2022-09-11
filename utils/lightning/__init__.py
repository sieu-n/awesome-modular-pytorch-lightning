import pytorch_lightning as pl


"""
# USE `trainer.estimated_stepping_batches` instead! - https://pytorch-lightning.readthedocs.io/en/stable/api/
# pytorch_lightning.trainer.trainer.Trainer.html#pytorch_lightning.trainer.trainer.Trainer.estimated_stepping_batches

def num_training_steps(trainer: pl.Trainer, pl_module: pl.LightningModule) -> int:
    # Total training steps inferred from datamodule and devices.
    dataset = pl_module.train_dataloader()
    if trainer.max_steps:
        return trainer.max_steps

    dataset_size = (
        trainer.limit_train_batches
        if trainer.limit_train_batches != 0
        else len(dataset)
    )

    num_devices = max(1, trainer.num_gpus, trainer.num_processes)
    if trainer.tpu_cores:
        num_devices = max(num_devices, trainer.tpu_cores)

    effective_batch_size = dataset.batch_size * trainer.accumulate_grad_batches * num_devices
    return (dataset_size // effective_batch_size) * trainer.max_epochs
"""