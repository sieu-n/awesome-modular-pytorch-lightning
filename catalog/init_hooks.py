# hook catalog
from utils.init_hooks.replace import SetModule, ResNetLowResHead # noqa

# utils
from ._get import _get


def get(name):
    return _get(globals(), name, "Initialization-hooks")
