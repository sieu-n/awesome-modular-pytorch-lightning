import torch
from torch import nn

from .merger import Merger
from .build import build_transforms
from lightning.base import _BaseLightningTrainer


class TTAFramework(_BaseLightningTrainer):
    """
    Wrap PyTorch nn.Module with test time augmentation transforms. Models can generally (e.g. classification,
    segmentation, ...) be wrapped using `TTAWrapper` assuming correct implementation of transforms.
    The augmentations can be weighted using a learning-based approach from the paper:
    Better Aggregation in Test-Time Augmentation, ICCV 2021

    Parameters
    ----------
    model: Union[torch.nn.Module, pl.LightningModule]
        model that has a `__call__` method that recieves x and predicts something.
    transforms: list[dict]
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
        num_classes=None,
        merge_mode="mean",
    ):
        super().__init__()
        self.predict_f = model.__call__    # don't store model to prevent recursion in state_dict.
        self.transforms = build_transforms(transforms)
        self.merge_mode = merge_mode
        self.output_key = output_label_key

        self.merger = Merger(type=self.merge_mode, n=len(self.transforms), num_classes=num_classes)

    def input_transform(self, x):
        # implement differently based on task, in: batch, out: input for model.__call__
        return x

    def process_augmented(self, x):
        # implement differently based on task, in: augmented sample
        return x

    def compute_result(self, x, pred):
        # implement differently based on task
        return pred

    def forward(self, x):
        """
        eval mode of TTA.
        """
        for i, transformer in enumerate(self.transforms):
            with torch.no_grad():
                augmented_x = transformer.augment_image(self.input_transform(x))
                augmented_output = self.predict_f(self.process_augmented(augmented_x))
                if self.output_key is not None:
                    augmented_output = augmented_output[self.output_key]
                deaugmented_output = transformer.deaugment_label(augmented_output)
            self.merger.update(x=deaugmented_output, i=i)

        agg_pred = self.merger.compute()
        self.merger.reset()
        return agg_pred, self.compute_result(x, agg_pred)

    def _training_step(self, batch, batch_idx=None):
        # train merger.
        agg_pred, res = self(batch)
        assert "loss" in res, "Loss must be returned to train TTA."
        return res["loss"], res

    def evaluate(self, batch, stage=None):
        agg_pred, res = self(batch)
        return res


class ClassificationTTAWrapper(TTAFramework):
    def __init__(self, model, output_label_key="logits", *args, **kwargs):
        # set default value of `output_label_key` to "logits"
        # This TTA wrapper is coupled with `lightning.vision.classification.ClassificationTrainer`
        self.classification_loss = model.classification_loss
        super().__init__(
            model=model,
            output_label_key=output_label_key,
            *args, **kwargs
        )

    def input_transform(self, x):
        # implement differently based on task
        return x["images"]

    def compute_result(self, x, logits):
        pred_prob = torch.nn.functional.log_softmax(logits, dim=1)
        d = {
            "logits": logits,
            "prob": pred_prob,
        }
        if "labels" in x:
            y = x["labels"]
            d["y"] = y
            d["cls_loss"] = self.classification_loss(logits, y)
            d["loss"] = self.classification_loss(logits, y)
        return d
