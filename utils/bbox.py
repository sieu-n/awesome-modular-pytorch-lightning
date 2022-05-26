import torch


def normalize_bbox(coord, img_w, img_h):
    # either [x1 y1 x2 y2] or [x y w h]
    return [coord[0] / img_w, coord[1] / img_h, coord[2] / img_w, coord[3] / img_h]


def unnormalize_bbox(coord, img_w, img_h):
    # either [x1 y1 x2 y2] or [x y w h]
    return [coord[0] * img_w, coord[1] * img_h, coord[2] * img_w, coord[3] * img_h]


def xywh_to_x1y1x2y2(xywh):
    return [xywh[0] - xywh[2], xywh[1] - xywh[3], xywh[0] + xywh[2], xywh[1] + xywh[3]]


def x1y1x2y2_to_xywh(x1y1x2y2):
    return [
        (x1y1x2y2[0] + x1y1x2y2[2]) / 2,
        (x1y1x2y2[1] + x1y1x2y2[3]) / 2,
        (x1y1x2y2[2] - x1y1x2y2[0]) / 2,
        (x1y1x2y2[3] - x1y1x2y2[1]) / 2,
    ]


def get_bbox_shape(anchor_size, aspect_ratio):
    return (anchor_size / aspect_ratio, anchor_size * aspect_ratio)


def get_bbox_shapes(anchor_sizes=[128, 256, 512], aspect_ratios=[0.5, 1.0, 2.0]):
    bbox_shapes = []
    for anchor_size in anchor_sizes:
        for aspect_ratio in aspect_ratios:
            bbox_shapes.append(get_bbox_shape(anchor_size, aspect_ratio))
    return bbox_shapes

# TODO,
# script the module to avoid hardcoded device type
@torch.jit.script_if_tracing
def _convert_boxes_to_pooler_format(
    boxes: torch.Tensor, sizes: torch.Tensor
) -> torch.Tensor:
    sizes = sizes.to(device=boxes.device)
    indices = torch.repeat_interleave(
        torch.arange(len(sizes), dtype=boxes.dtype, device=boxes.device), sizes
    )
    return cat([indices[:, None], boxes], dim=1)


def convert_boxes_to_pooler_format(box_lists):
    """
    Convert all boxes in `box_lists` to the low-level format used by ROI pooling ops
    (see description under Returns).
    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]):
            A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
    Returns:
        When input is list[Boxes]:
            A tensor of shape (M, 5), where M is the total number of boxes aggregated over all
            N batch images.
            The 5 columns are (batch index, x0, y0, x1, y1), where batch index
            is the index in [0, N) identifying which batch image the box with corners at
            (x0, y0, x1, y1) comes from.
        When input is list[RotatedBoxes]:
            A tensor of shape (M, 6), where M is the total number of boxes aggregated over all
            N batch images.
            The 6 columns are (batch index, x_ctr, y_ctr, width, height, angle_degrees),
            where batch index is the index in [0, N) identifying which batch image the
            rotated box (x_ctr, y_ctr, width, height, angle_degrees) comes from.
    """
    boxes = torch.cat([x.tensor for x in box_lists], dim=0)
    # __len__ returns Tensor in tracing.
    sizes = shapes_to_tensor([x.__len__() for x in box_lists])
    return _convert_boxes_to_pooler_format(boxes, sizes)
