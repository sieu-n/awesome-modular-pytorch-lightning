import torchmetrics


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
