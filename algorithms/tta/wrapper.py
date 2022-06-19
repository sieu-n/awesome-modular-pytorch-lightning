import torch
from torch import nn

from .merger import Merger
from .build import build_transforms


class TTAWrapper(nn.Module):
    """
    Wrap PyTorch nn.Module with test time augmentation transforms. Models can generally (e.g. classification,
    segmentation, ...) be wrapped using `TTAWrapper` assuming correct implementation of transforms.
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
        merge_mode="mean",
    ):
        super().__init__()
        self.predict_f = model.__call__    # don't store model to prevent recursion in state_dict.
        self.transforms = build_transforms(transforms)
        self.merge_mode = merge_mode
        self.output_key = output_label_key

        self.merger = Merger(type=self.merge_mode, n=len(self.transforms))

    def input_transform(self, x):
        # implement differently based on task, in: batch, out: input for model.__call__
        return x

    def compute_result(self, x, pred):
        # implement differently based on task
        return pred

    def forward(self, x):
        """
        eval mode of TTA.
        """
        for transformer in self.transforms:
            augmented_x = transformer.augment_image(self.input_transform(x))
            augmented_output = self.predict_f(augmented_x)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.deaugment_label(augmented_output)
            self.merger.update(deaugmented_output)

        agg_pred = self.merger.compute()
        self.merger.reset()
        return agg_pred, self.compute_result(x, agg_pred)


class ClassificationTTAWrapper(TTAWrapper):
    def __init__(self, model, output_label_key="logits", *args, **kwargs):
        # set default value of `output_key` to "logits"
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
        return d
