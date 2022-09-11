# Implement gradient noise using the method described in the paper:
# Adding Gradient Noise Improves Learning for Very Deep Networks

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class AdditiveGradientNoiseCallback(Callback):
    def __init__(self, eta, log_sigma=True):
        self.eta = eta
        self.log_sigma = log_sigma

    def get_sigma(self):
        return self.eta

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx):
        sigma = self.get_sigma()
        if self.log_sigma:
            pl_module.log("step/noise-sigma", sigma)

        for param in pl_module.parameters():
            if param.grad is not None:
                param.grad += torch.randn_like(param.grad) * sigma


class DecayingGradientNoiseCallback(Callback):
    def __init__(self, eta: float, gamma: float = 0.55, log_sigma: bool = True):
        self.eta = eta
        self.gamma = gamma
        self.log_sigma = log_sigma

    def get_sigma(self, step: int):
        return self.eta / ((1 + step) ** self.gamma)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx):
        sigma = self.get_sigma(trainer.global_step)
        if self.log_sigma:
            pl_module.log("step/noise-sigma", sigma)

        for param in pl_module.parameters():
            if param.grad is not None:
                param.grad += torch.randn_like(param.grad) * sigma
