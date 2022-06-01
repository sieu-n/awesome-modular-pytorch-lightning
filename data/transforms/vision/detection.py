import random
import torch

import torchvision.transforms.functional as TF
from data.transforms.base import _BaseTransform
from utils.bbox import (
    normalize_bbox,
    unnormalize_bbox,
    x1y1x2y2_to_xywh,
    xywh_to_x1y1x2y2,
)


class DetectionVOCLabelTransform(_BaseTransform):
    def __init__(self, **kwargs):
        """
        Recieves the naive `VOC2012` torchvision dataset, which contains the annotations from the `.xml` file.
        Processes and returns the class and bbox in the following format:

        x = PIL.Image
        y = {"boxes": list[list[x, y, w, h], ...], "labels": list[int]}

        Each bbox coordinates are given in (x, y, w, h) format. The numbers are normalized to (0, 1) range by dividing
        them with the width and height of the image.
        """
        super().__init__(**kwargs)
        self.label2code = {
            name: idx for idx, name in enumerate(self.const_cfg["label_map"])
        }

    def __call__(self, x, y):
        boxes, labels = self.transform(y, x.size(2), x.size(1))
        return {"images": x, "boxes": boxes, "labels": labels}

    def transform(self, label, img_w, img_h):
        """
        img_w: int
        img_h: int
        label: dict
            output of VOC-style .xml annotation
        """
        label = label["annotation"]["object"]
        targets = {"boxes": [], "labels": []}
        for obj_label in label:
            bbox = obj_label["bndbox"]
            x1, y1, x2, y2 = (
                int(bbox["xmin"]),
                int(bbox["ymin"]),
                int(bbox["xmax"]),
                int(bbox["ymax"]),
            )
            bbox_xywh = x1y1x2y2_to_xywh([x1, y1, x2, y2])
            targets["boxes"].append(normalize_bbox(bbox_xywh, img_w, img_h))
            targets["labels"].append(self.label2code[obj_label["name"]])

        # returns {"boxes": list[list[x, y, w, h], ...], "labels": list[int]}
        return torch.tensor(targets["boxes"]), torch.tensor(targets["labels"])


class YOLObbox2Pytorch(_BaseTransform):
    def __call__(self, d):
        d["boxes"] = self.transform(d["boxes"], d["images"].size(2), d["images"].size(1))
        return d

    def transform(self, boxes, img_w, img_h):
        """
        Convert YOLO-format relative (x, y, w, h) bbox to PyTorch-style absolute (x1, y1, x2, y2) coordinates.
        Parameters
        ----------
        img_w: int
        img_h: int
        boxes: list[list[x, y, w, h]]
        """
        return unnormalize_bbox(xywh_to_x1y1x2y2(boxes), img_w, img_h)


class Pytorchbbox2YOLO(_BaseTransform):
    def __call__(self, d):
        d["boxes"] = self.transform(d["boxes"], d["images"].size(2), d["images"].size(1))
        return d

    def transform(self, boxes, img_w, img_h):
        """
        Convert PyTorch-style absolute (x1, y1, x2, y2) bbox to YOLO-format relative (x, y, w, h) coordinates.
        Parameters
        ----------
        img_w: int
        img_h: int
        boxes: list[list[x1, y1, x2, y2]]
        """
        return normalize_bbox(x1y1x2y2_to_xywh(boxes), img_w, img_h)


class DetectionCropToRatio(_BaseTransform):
    def __init__(self, max_ratio=1, min_ratio=1, mode="center", **kwargs):
        super().__init__(**kwargs)
        assert mode in ("center", "random")
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.mode = mode

    def __call__(self, d):
        # assert type(d) == dict and len(d.keys()) == 3
        # assert "images" in d
        # assert "boxes" in d
        # assert "labels" in d

        cropped_image, shifted_boxes, is_removed = self.transform(d["image"], d["boxes"])
        num_boxes = len(is_removed)
        box_mask = (torch.tensor(is_removed) == False)

        new_data = {"images": cropped_image, "boxes": shifted_boxes}
        for key in d.keys():
            if key == "images" or key == "boxes":
                continue
            if hasattr(d[key], '__len__') and len(d[key]) == num_boxes:
                # if some attribute is assigned per-box such as `label`, remove some of them.
                new_data[key] = d[key][box_mask]
            else:
                new_data[key] = d[key]
        return new_data

    def transform(self, image, boxes):
        """
        image: torch.Tensor (C, H, W)
        label: list[dict] - [{"boxes": [x, y, w, h], "cls": str}, ...]
        """
        h, w = image.size(1), image.size(2)
        ratio = h / w
        w_min, w_max = 0, w
        h_min, h_max = 0, h

        if ratio > self.max_ratio:
            # crop along height
            target_h = int(w * self.max_ratio)
            if self.mode == "center":
                pad_h = (h - target_h) // 2
                h_min, h_max = pad_h, pad_h + target_h
            elif self.mode == "random":
                pad_h = random.randint(0, h - target_h)
                h_min, h_max = pad_h, pad_h + target_h
        elif ratio < self.min_ratio:
            # crop along width
            target_w = int(h / self.min_ratio)
            if self.mode == "center":
                pad_w = (w - target_w) // 2
                w_min, w_max = pad_w, pad_w + target_w
            elif self.mode == "random":
                pad_w = random.randint(0, w - target_w)
                w_min, w_max = pad_w, pad_w + target_w

        # crop image
        cropped_image = image[:, h_min:h_max, w_min:w_max]
        # move bbox(given in relative [x, y, w, h] coordinates)
        shifted_boxes = []
        is_removed = [False] * len(boxes)

        for obj_idx in range(len(boxes)):
            x1, y1, x2, y2 = unnormalize_bbox(
                xywh_to_x1y1x2y2(boxes[obj_idx]), w, h
            )
            # check if bbox is outside cropped image
            if x1 >= w_max or x2 <= w_min or y1 >= h_max or y2 < h_min:
                is_removed[obj_idx] = True
                continue
            # clip coords inside bbox.
            x1, y1, x2, y2 = (
                max(x1, w_min),
                max(y1, h_min),
                min(x2, w_max),
                min(y2, h_max),
            )
            # shift bbox
            x1, x2 = x1 - w_min, x2 - w_min
            y1, y2 = y1 - h_min, y2 - h_min
            new_bbox = normalize_bbox(
                x1y1x2y2_to_xywh([x1, y1, x2, y2]), w_max - w_min, h_max - h_min
            )
            shifted_boxes.append(new_bbox)

        return cropped_image, torch.tensor(shifted_boxes), is_removed


class DetectionConstrainImageSize(_BaseTransform):
    # https://github.com/facebookresearch/maskrcnn-benchmark/blob/main/maskrcnn_benchmark/data/transforms/transforms.py
    def __init__(self, min_size, max_size, **kwargs):
        """
        Constrain image size so that smaller side is larger than `min_size` and larger side is smaller than `max_size`.
        """
        super().__init__(**kwargs)
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, d):
        d["image"] = self.transform(d["image"])
        return d

    # modified from torchvision to add support for max size
    def get_size(self, w, h):
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def transform(self, image):
        """
        image: torch.Tensor (C, H, W)
        label: list[dict] - [{"boxes": [x, y, w, h], "cls": str}, ...]
        """
        size = self.get_size(image.size(2), image.size(1))
        image = TF.resize(image, size)
        # no need to transform labels as they are already normalized
        # target = target.resize(image.size)
        return image

################################################################
# Data Augmentation for object detection
################################################################


class DetectionHFlip(_BaseTransform):
    def __init__(self, prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob

    def __call__(self, d):
        if random.random() < self.prob:
            d["image"] = TF.hflip(d["image"])
            d["boxes"][0] = 1.0 - d["boxes"][0]
        return d


class DetectionVFlip(_BaseTransform):
    def __init__(self, prob=0.5, **kwargs):
        super(self).__init__(**kwargs)
        self.prob = prob

    def __call__(self, d):
        if random.random() < self.prob:
            d["image"] = TF.vflip(d["image"])
            d["boxes"][1] = 1.0 - d["boxes"][1]
        return d
