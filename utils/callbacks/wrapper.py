from pytorch_lightning import callbacks as _PytorchLightningCallbacks


def LightningCallback(name, args={}):
    return getattr(_PytorchLightningCallbacks, name)(**args)
