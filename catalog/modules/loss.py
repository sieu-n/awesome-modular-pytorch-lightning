from timm.loss import BinaryCrossEntropy as OneToAllBinaryCrossEntropy  # noqa E403

from algorithms.loss.classification import SoftTargetCrossEntropy  # noqa E403
from algorithms.loss.focal import CohenKappaWeight  # noqa E403
from algorithms.loss.polyloss import *  # noqa F401
