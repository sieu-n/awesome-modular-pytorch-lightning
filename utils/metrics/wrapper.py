import torchmetrics


def TorchMetric(name, args={}):
    return getattr(torchmetrics, name)(**args)
