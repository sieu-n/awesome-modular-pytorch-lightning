# Implement gradient noise using the method described in the paper:
# Adding Gradient Noise Improves Learning for Very Deep Networks

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class DecayingGradientNoiseCallback(Callback):
    def __init__(self, eta, gamma):
        pass

    def get_sigma(self):
        
    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx):
        sigma = self.get_sigma()