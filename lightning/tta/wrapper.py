import torch
from torch import nn
from ttach.base import Merger
import pytorch_lightning as pl

from lightning.base import _BaseLightningTrainer


class TTAWrapper(_BaseLightningTrainer):
    """
    Wrap PyTorch nn.Module with test time augmentation transforms. Models can generally (e.g. classification, 
    segmentation, ...) be wrapped using `TTAWrapper` assuming correct implementation of transforms.
    Parameters
    ----------
    model: Union[torch.nn.Module, pl.LightningModule]
        model that has a `__call__` method that recieves x and predicts something.
    transforms: list[object]
        Composition of test time transforms
    merge_mode: str
        method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
    output_label_key: str
        if model output is `dict`, specify which key belong to `label`
    """

    def __init__(
        self,
        model,
        transforms,
        output_label_key=None,
        merge_mode="mean",
    ):
        super().__init__()
        self.register_buffer('model', model)
        self.transforms = transforms
        self.merge_mode = merge_mode

    def _training_step():
        raise ValueError("`ClassificationTTAWrapper` should only be used for validation and prediction.")

    def forward(
        self, image: torch.Tensor, *args
    ):
        merger = Merger(type=self.merge_mode, n=len(self.transforms))

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image, *args)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.deaugment_label(augmented_output)
            merger.append(deaugmented_output)

        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}

        return result
