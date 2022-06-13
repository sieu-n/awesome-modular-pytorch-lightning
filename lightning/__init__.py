from . import base, trainers, vision  # noqa F401


########################################################################
# Pytorch-lightning utils.
########################################################################
def get(name):
    return getattr(trainers, name)
