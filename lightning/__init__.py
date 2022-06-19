from . import base, catalog, vision  # noqa F401


########################################################################
# Pytorch-lightning utils.
########################################################################
def get(name):
    return getattr(catalog, name)
