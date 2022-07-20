try:
    from mmcv.parallel.data_container import DataContainer
except ImportError:
    pass

from data.transforms.base import _BaseTransform
from utils.bbox import (
    normalize_bbox,
    unnormalize_bbox,
    x1y1x2y2_to_xywh,
    xywh_to_x1y1x2y2,
)


class YOLObbox2Pytorch(_BaseTransform):
    """
    Convert bounding box format from relative to absolute.
    """
    def __call__(self, d):
        if len(d["boxes"]) > 0:
            d["boxes"] = self.transform(
                d["boxes"], d["images"].size(2), d["images"].size(1)
            )
        return d

    def transform(self, boxes, img_w, img_h):
        """
        Convert YOLO-format relative (x, y, w, h) bbox to PyTorch-style absolute (x1, y1, x2, y2) coordinates.
        Parameters
        ----------
        img_w: int
        img_h: int
        boxes: torch.Tensor(4, num_obj)
        """
        return unnormalize_bbox(xywh_to_x1y1x2y2(boxes), img_w, img_h)


class Pytorchbbox2YOLO(_BaseTransform):
    """
    Convert bounding box format from relative to absolute.
    """
    def __call__(self, d):
        if len(d["boxes"]) > 0:
            d["boxes"] = self.transform(
                d["boxes"], d["images"].size(2), d["images"].size(1)
            )
        return d

    def transform(self, boxes, img_w, img_h):
        """
        Convert PyTorch-style absolute (x1, y1, x2, y2) bbox to YOLO-format relative (x, y, w, h) coordinates.
        Parameters
        ----------
        img_w: int
        img_h: int
        boxes: torch.Tensor(4, num_obj)
        """
        return normalize_bbox(x1y1x2y2_to_xywh(boxes), img_w, img_h)


class MMdetDataset2Torchvision(_BaseTransform):
    """
    Convert MMdetection style dataset to default format(see README).
    Remove wrapper if data is wrapped in `mmcv.parallel.data_container.DataContainer`.
    Parameters
    ----------
    to_xywh: bool
        If true, convert to YOLO-format relative (x, y, w, h) bounding box. If false, only change key and maintain
        pascal transform format.
    """
    def __init__(self, to_xywh=True, *args, **kwargs):
        self.to_xywh = to_xywh
        super().__init__(*args, **kwargs)

    def __call__(self, d):
        # initially map data to new key.
        torchvision_d = {
            "images": d["img"],
            "boxes": d["gt_bboxes"],
            "labels": d["gt_labels"],
            "meta": d["img_metas"],
        }
        # remove DataContainer by calling v.data
        torchvision_d = {k: (v.data if isinstance(v, DataContainer) else v) for k, v in d.items()}
        # (optional) transform bounding box format.
        if self.to_xywh:
            if "img_shape" in d["meta"]:
                w, h = d["meta"][0], d["meta"][1]
            else:
                w, h = d["img"].size(2), d["img"].size(1)
            torchvision_d["boxes"] = normalize_bbox(x1y1x2y2_to_xywh(torchvision_d["boxes"]), w, h)
        return torchvision_d
