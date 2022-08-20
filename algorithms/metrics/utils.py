from typing import Iterable

import catalog.metric
import numpy as np
import torch
import torchmetrics
from torch import nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def TorchMetric(name, args={}):
    return getattr(torchmetrics, name)(**args)


class SubsetMetric(torchmetrics.Metric):
    """
    Separately measure metric for each subset of the dataset.

    Parameters
    ----------
    name: str
        Name of base metric.
    args: dict
        Arguments for the base metric.
    num_subsets: int
        Number of subsets that separate the dataset.
    get_avg: bool, default: False
        Returns the average of all the subsets if specified.
    """

    def __init__(
        self, name: str, num_subsets: int, args: dict = {}, get_avg: bool = False
    ):
        super().__init__()

        self.metrics = nn.ModuleList(
            [catalog.metric.build(name=name, args=args) for _ in range(num_subsets)]
        )
        self.name2idx = {}
        self.get_avg = get_avg

    def update(self, subset: Iterable, *args, **kwargs):
        for s in subset:
            if s not in self.name2idx:
                self.name2idx[s] = len(self.name2idx)
            idx = self.name2idx[s]

            self.metrics[idx].update(*args, **kwargs)

    def compute(self):
        r = {
            name: self.metrics[self.name2idx[name]].compute()
            for name, idx in self.name2idx.items()
        }
        if self.get_avg:
            avg_res = sum(r.values()) / len(r)
            r.update({"AVERAGE": avg_res})
        return r

    def reset(self):
        for m in self.metrics:
            m.reset()


class MMDet2TorchMetricmAP(MeanAveragePrecision):
    def update(self, pred_boxes, pred_scores, target_boxes, target_labels):
        labels = []
        for idx in range(len(pred_scores)):
            labels += [idx + 1] * len(pred_scores[idx])
        super(MMDet2TorchMetricmAP, self).update(
            preds=[
                dict(
                    boxes=torch.tensor(np.concatenate(pred_boxes)),
                    scores=torch.tensor(np.concatenate(pred_scores)),
                    labels=torch.tensor(labels),
                )
            ],
            target=[
                dict(
                    boxes=target_boxes,
                    labels=target_labels,
                )
            ],
        )
