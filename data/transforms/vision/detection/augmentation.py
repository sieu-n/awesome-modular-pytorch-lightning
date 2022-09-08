import random

import torch
import torchvision.transforms.functional as TF
from ...utils import _BaseTransform
from utils.bbox import (
    normalize_bbox,
    unnormalize_bbox,
    x1y1x2y2_to_xywh,
    xywh_to_x1y1x2y2,
)


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
        is_removed = [True]
        cnt = 0
        while sum(is_removed) == len(is_removed):  # loop until at least one box is left
            cropped_image, shifted_boxes, is_removed = self.transform(
                d["images"], d["boxes"]
            )
            cnt += 1
            if cnt == 10:  # give up after 10 iterations.
                return d

        num_boxes = len(is_removed)
        box_mask = torch.tensor(is_removed) == False  # noqa E712

        new_data = {"images": cropped_image, "boxes": shifted_boxes}
        for key in d.keys():
            if key == "images" or key == "boxes":
                continue
            if hasattr(d[key], "__len__") and len(d[key]) == num_boxes:
                # if some attribute is assigned per-box such as `label`, remove some of them.
                new_data[key] = d[key][box_mask]
            else:
                new_data[key] = d[key]
        return new_data

    def transform(self, image, boxes):
        """
        image: torch.Tensor (C, H, W)
        boxes: torch.Tensor(4, num_obj)
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
            x1, y1, x2, y2 = unnormalize_bbox(xywh_to_x1y1x2y2(boxes[obj_idx]), w, h)
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
        d["images"] = self.transform(d["images"])
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
            d["images"] = TF.hflip(d["images"])
            if len(d["boxes"]) > 0:
                d["boxes"][..., 0] = 1.0 - d["boxes"][..., 0]
        return d


class DetectionVFlip(_BaseTransform):
    def __init__(self, prob=0.5, **kwargs):
        super(self).__init__(**kwargs)
        self.prob = prob

    def __call__(self, d):
        if random.random() < self.prob:
            d["images"] = TF.vflip(d["images"])
            if len(d["boxes"]) > 0:
                d["boxes"][..., 1] = 1.0 - d["boxes"][..., 1]
        return d
