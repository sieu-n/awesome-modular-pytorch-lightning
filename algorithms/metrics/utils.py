import torchmetrics
import catalog.metric
from torch import nn


def TorchMetric(name, args={}):
    return getattr(torchmetrics, name)(**args)


class SubsetMetric(torchmetrics.Metric):
    """
    Separately measure metric for each subset of the dataset.
    """
    def __init__(self, name: str, num_subsets: int, args: dict = {}, get_avg: bool = False):
        super().__init__()

        self.metrics = nn.ModuleList([
            catalog.metric.build(name=name, args=args) for _ in range(num_subsets)
        ])
        self.name2idx = {}
        self.get_avg = get_avg

    def update(self, subset: str, *args, **kwargs):
        if subset not in self.name2idx:
            self.name2idx[subset] = len(self.name2idx)
        idx = self.name2idx[subset]
        return self.metrics[idx].update(*args, **kwargs)

    def compute(self):
        r = {name: self.metrics[self.name2idx[name]].compute()
             for name, idx in self.name2idx.items()}
        if self.get_avg:
            avg_res = sum(r.values()) / len(r)
            r.update({"AVERAGE": avg_res})
        return r

    def reset(self):
        for m in self.metrics:
            m.reset()
