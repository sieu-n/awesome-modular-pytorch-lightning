# hook catalog
from utils.init_hooks.replace import (  # noqa
    ReplaceModulesOfType,
    ResNetLowResHead,
    SetModule,
)

# utils
from ._get import _get


def get(name):
    return _get(globals(), name, "Initialization-hooks")
