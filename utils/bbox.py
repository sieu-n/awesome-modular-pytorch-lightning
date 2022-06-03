import torch


def stack_and_round_boxes(x):
    return torch.stack(x, dim=-1)


def normalize_bbox(coord, img_w, img_h):
    """
    Parameters
    ----------
    coord : np.array: [*, 4], torch.tensor: [*, 4], or list[*, 4]
    img_w : float or int
    img_h : float or int
    """
    input_type = type(coord)
    # either [x1 y1 x2 y2] or [x y w h] format, either 1d or 2d [[x1 y1 x2 y2], ...]
    if not isinstance(coord, torch.Tensor):
        coord = torch.tensor(coord)
    return input_type(stack_and_round_boxes([
        coord[..., 0] / img_w,
        coord[..., 1] / img_h,
        coord[..., 2] / img_w,
        coord[..., 3] / img_h,
    ]))


def unnormalize_bbox(coord, img_w, img_h):
    """
    Parameters
    ----------
    coord : np.array: [*, 4], torch.tensor: [*, 4], or list[*, 4]
    img_w : float or int
    img_h : float or int
    """
    input_type = type(coord)
    # either [x1 y1 x2 y2] or [x y w h] format, either 1d or 2d: shape(4,) [[x1 y1 x2 y2], ...]
    if not isinstance(coord, torch.Tensor):
        coord = torch.tensor(coord)
    return input_type(stack_and_round_boxes([
        coord[..., 0] * img_w,
        coord[..., 1] * img_h,
        coord[..., 2] * img_w,
        coord[..., 3] * img_h,
    ]))


def xywh_to_x1y1x2y2(xywh):
    """
    Parameters
    ----------
    xywh : np.array: [*, 4], torch.tensor: [*, 4], or list[*, 4]
    """
    input_type = type(xywh)
    if not isinstance(xywh, torch.Tensor):
        xywh = torch.tensor(xywh)
    return input_type(stack_and_round_boxes([
        xywh[..., 0] - xywh[..., 2] / 2,
        xywh[..., 1] - xywh[..., 3] / 2,
        xywh[..., 0] + xywh[..., 2] / 2,
        xywh[..., 1] + xywh[..., 3] / 2,
    ]))


def x1y1x2y2_to_xywh(x1y1x2y2):
    """
    Parameters
    ----------
    x1y1x2y2 : np.array: [*, 4], torch.tensor: [*, 4], or list[*, 4]
    """
    input_type = type(x1y1x2y2)
    if not isinstance(x1y1x2y2, torch.Tensor):
        x1y1x2y2 = torch.tensor(x1y1x2y2)
    return input_type(stack_and_round_boxes([
        (x1y1x2y2[..., 0] + x1y1x2y2[..., 2]) / 2,
        (x1y1x2y2[..., 1] + x1y1x2y2[..., 3]) / 2,
        x1y1x2y2[..., 2] - x1y1x2y2[..., 0],
        x1y1x2y2[..., 3] - x1y1x2y2[..., 1],
    ]))


def check_isvalid_boxes(batch_of_boxes, img_w=1.0, img_h=1.0, xywh=True, is_batch=True, tolerance=0.001):
    """
    Check if there is invalid boxes. For example, no boxes should have coordinates smaller that 0 or greater than the
    width / height of the image.
    Parameters
    ----------
    boxes: list[torch.Tensor(num_obj, 4)]
        bounding boxes. The format should be consistent with the other arguments.
    w: int, float, or list[int, float]
    h: int, float, or list[int, float]
    xywh: bool
        Whether coordinates of the bbox are given in [x, y, w, h] format(YOLO format). If False, it is consider to be
        in [x1, y1, x2, y2] format.
    is_batch: bool
    """
    if not is_batch:  # add fake bbox to verify single set of bbox.
        batch_of_boxes = [batch_of_boxes]
    is_imsize_list = False
    w, h = img_w, img_h
    if hasattr(img_w, "__len__") or hasattr(img_h, "__len__"):
        assert hasattr(img_w, "__len__") and hasattr(img_h, "__len__")
        assert len(img_w) == len(img_h) == len(batch_of_boxes)
        is_imsize_list = True

    for idx, boxes in enumerate(batch_of_boxes):
        # if `boxes` is empty(no objects are present)
        if len(boxes) == 0:
            continue

        if is_imsize_list:
            w, h = img_w[idx], img_h[idx]
        w_tolerance, h_tolerance = w * tolerance, h * tolerance

        # 1. check dimensions.
        assert isinstance(boxes, torch.Tensor) and boxes.shape[1] == 4
        # 2. check obvious things.
        if xywh:
            # width and height should all be greater than 0
            assert torch.min(boxes[..., 2]) >= 0.0 and torch.min(boxes[..., 3]) >= 0.0
        else:
            # check if y2 >= y1 and x2 >= x1.
            assert (
                torch.min(boxes[..., 2] - boxes[..., 0]) >= 0.0
                and torch.min(boxes[..., 3] - boxes[..., 1]) >= 0.0
            )
        # 3. check if some boxes have coordinates outside the image
        if xywh:
            boxes = xywh_to_x1y1x2y2(boxes)
        assert (torch.min(boxes[..., 0]) >= 0.0 - w_tolerance and
                torch.min(boxes[..., 1]) >= 0.0 - h_tolerance), f"Got {boxes}"
        assert (torch.max(boxes[..., 2]) <= w + w_tolerance and
                torch.max(boxes[..., 3]) <= h + h_tolerance), f"Got {boxes}"


def get_anchor_shape(anchor_size, aspect_ratio):
    return (anchor_size / aspect_ratio, anchor_size * aspect_ratio)


def get_anchor_shapes(anchor_sizes=[128, 256, 512], aspect_ratios=[0.5, 1.0, 2.0]):
    bbox_shapes = []
    for anchor_size in anchor_sizes:
        for aspect_ratio in aspect_ratios:
            bbox_shapes.append(get_anchor_shape(anchor_size, aspect_ratio))
    return bbox_shapes
