from ttach.transforms import Add  # noqa E403; Add value to images
from ttach.transforms import Multiply  # noqa E403; Multiply images by factor
from ttach.transforms import Resize  # noqa E403; Resize images
from ttach.transforms import Rotate90  # noqa E403; Rotate images 0/90/180/270 degrees
from ttach.transforms import Scale  # noqa E403; Scale images
from ttach.transforms import (  # noqa E403; Makes 4 crops for each corner + center crop; noqa E403; Flip images horizontally (left->right); noqa E403; Flip images vertically (up->down)
    FiveCrops,
    HorizontalFlip,
    VerticalFlip,
    FiveCrops,
)

from .affine import CenterZoom  # noqa E403; Rotate image with arbitrary angles.
from .affine import Rotation  # noqa E403; Zoom into center of the image.
from .affine import FiveCrops  # noqa E403; Zoom into center of the image.
