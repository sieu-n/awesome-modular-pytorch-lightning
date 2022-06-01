import torch


# TODO refactor to further improve polymorphism.
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
    return input_type(torch.stack([
        coord[..., 0] / img_w,
        coord[..., 1] / img_h,
        coord[..., 2] / img_w,
        coord[..., 3] / img_h
    ], dim=-1))


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
    return input_type(torch.stack([
        coord[..., 0] * img_w,
        coord[..., 1] * img_h,
        coord[..., 2] * img_w,
        coord[..., 3] * img_h
    ], dim=-1))


def xywh_to_x1y1x2y2(xywh):
    """
    Parameters
    ----------
    xywh : np.array: [*, 4], torch.tensor: [*, 4], or list[*, 4]
    """
    input_type = type(xywh)
    if not isinstance(xywh, torch.Tensor):
        xywh = torch.tensor(xywh)
    return input_type(torch.stack([
        xywh[..., 0] - xywh[..., 2],
        xywh[..., 1] - xywh[..., 3],
        xywh[..., 0] + xywh[..., 2],
        xywh[..., 1] + xywh[..., 3]
    ], dim=-1))


def x1y1x2y2_to_xywh(x1y1x2y2):
    """
    Parameters
    ----------
    x1y1x2y2 : np.array: [*, 4], torch.tensor: [*, 4], or list[*, 4]
    """
    input_type = type(x1y1x2y2)
    if not isinstance(x1y1x2y2, torch.Tensor):
        x1y1x2y2 = torch.tensor(x1y1x2y2)
    return input_type(torch.stack([
        (x1y1x2y2[..., 0] + x1y1x2y2[..., 2]) / 2,
        (x1y1x2y2[..., 1] + x1y1x2y2[..., 3]) / 2,
        (x1y1x2y2[..., 2] - x1y1x2y2[..., 0]) / 2,
        (x1y1x2y2[..., 3] - x1y1x2y2[..., 1]) / 2,
    ], dim=-1))


def get_anchor_shape(anchor_size, aspect_ratio):
    return (anchor_size / aspect_ratio, anchor_size * aspect_ratio)


def get_anchor_shapes(anchor_sizes=[128, 256, 512], aspect_ratios=[0.5, 1.0, 2.0]):
    bbox_shapes = []
    for anchor_size in anchor_sizes:
        for aspect_ratio in aspect_ratios:
            bbox_shapes.append(get_anchor_shape(anchor_size, aspect_ratio))
    return bbox_shapes
