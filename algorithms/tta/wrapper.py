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
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_label_key

        self.merger = Merger(type=self.merge_mode, n=len(self.transforms))

    def label_transform(self, label):
        # implement differently based on task
        return label

    def forward(self, x):
        """
        eval mode of TTA.
        """
        for transformer in self.transforms:
            augmented_x = transformer.augment_image(x)
            augmented_output = self.model(augmented_x)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.deaugment_label(augmented_output)
            self.merger.update(deaugmented_output)

        result = self.merger.compute()
        self.merger.reset()
        if self.output_key is not None:
            result = {self.output_key: result}

        return result


class ClassificationTTAWrapper(TTAWrapper):
    def label_transform(self, logits):
        pred_prob = torch.nn.functional.log_softmax(logits, dim=1)
        return {
            "logits": logits,
            "prob": pred_prob,
        }
