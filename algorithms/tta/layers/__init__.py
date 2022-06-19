from ttach.transforms import (  # noqa E403
    HorizontalFlip,             # Flip images horizontally (left->right)
    VerticalFlip,               # Flip images vertically (up->down)
    Rotate90,                   # Rotate images 0/90/180/270 degrees
    Scale,                      # Scale images
    Resize,                     # Resize images
    Add,                        # Add value to images
    Multiply,                   # Multiply images by factor
    FiveCrops,                  # Makes 4 crops for each corner + center crop
)
