+# Implement gradient noise using the method described in the paper:
# Adding Gradient Noise Improves Learning for Very Deep Networks

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class AdditiveGradientNoiseCallback(Callback):
    def __init__(self, eta):
        self.eta = eta

    def get_sigma(self):
        return self.eta

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx):
        sigma = self.get_sigma()
        for param in pl_module.parameters():
            param.grad += torch.randn_like(param.grad) * sigma


class DecayingGradientNoiseCallback(Callback):
    def __init__(self, eta, gamma=0.55):
        self.eta = eta
        self.gamma = gamma

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = self.train_dataloader()
        if self.trainer.max_steps:
            return self.trainer.max_steps

        dataset_size = (
            self.trainer.limit_train_batches
            if self.trainer.limit_train_batches != 0
            else len(dataset)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
        return (dataset_size // effective_batch_size) * self.trainer.max_epochs

    def get_sigma(self):
        return self.eta / ((1 + self.num_training_steps) ** self.gamma)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx):
        sigma = self.get_sigma()
        for param in pl_module.parameters():
            param.grad += torch.randn_like(param.grad) * sigma
