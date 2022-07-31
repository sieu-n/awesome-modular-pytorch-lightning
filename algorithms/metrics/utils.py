import torchmetrics
import catalog.metric
from torch import nn


class TorchMetric(torchmetrics.Metric):
    def __init__(self, NAME: str, ARGS: dict):
        """
        Wrapper for using metrics that are implemented in torchmetrics.
        Parameters
        ----------
        NAME: str
            Exact name of the metric in torchmetrics.
        ARGS: dict
            Dictionary of arguments to pass when constructing the metric.
        """
        super().__init__()
        self.metric = getattr(torchmetrics, NAME)(**ARGS)

    def update(self, *args, **kwargs):
        return self.metric.update(*args, **kwargs)

    def compute(self, *args, **kwargs):
        return self.metric.compute(*args, **kwargs)

    def reset(self, *args, **kwargs):
        return self.metric.reset(*args, **kwargs)


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
