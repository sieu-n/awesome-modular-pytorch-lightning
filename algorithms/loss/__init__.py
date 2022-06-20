from timm.loss import BinaryCrossEntropy as OneToAllBinaryCrossEntropy  # noqa E403

from .classification import SoftTargetCrossEntropy  # noqa E403
from .focal import CohenKappaWeight  # noqa E403
from .polyloss import *  # noqa F401
