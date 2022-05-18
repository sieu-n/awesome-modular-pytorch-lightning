import random

import torchvision.transforms.functional as TF
from data.transforms.common import _BaseTransform
from utils.bbox import pixel_bbox_to_relative, x1y1x2y2_to_xywh, xywh_to_x1y1x2y2


class DetectionVOCLabelTransform(_BaseTransform):
    def __init__(self, **kwargs):
        """
        Recieves the naive `VOC2012` torchvision dataset, which contains the annotations from the `.xml` file.
        Processes and returns the class and bbox in the following format:

        x = PIL.Image
        y = [{"bbox": list[x, y, w, h], "class": int} ]

        Each bbox coordinates are given in (x, y, w, h) format. The numbers are normalized to (0, 1) range by dividing
        them with the width and height of the image.
        """
        super().__init__(**kwargs)
        self.label2code = {
            name: idx for idx, name in enumerate(self.const_cfg["label_map"])
        }

    def joint_transform(self, image, label):
        """
        image: torch.Tensor (C, H, W)
        label: dict - output of VOC-style .xml annotation
        """
        img_w, img_h = image.size(2), image.size(1)
        label = label["annotation"]["object"]
        targets = []
        for obj_label in label:
            bbox = obj_label["bndbox"]
            x1, y1, x2, y2 = (
                int(bbox["xmin"]),
                int(bbox["ymin"]),
                int(bbox["xmax"]),
                int(bbox["ymax"]),
            )
            bbox_xywh = x1y1x2y2_to_xywh([x1, y1, x2, y2])
            targets.append(
                {
                    "bbox": pixel_bbox_to_relative(bbox_xywh, img_w, img_h),
                    "class": self.label2code[obj_label["name"]],
                }
            )
        # returns [{"bbox": [x, y, w, h], "cls": str}, ...]
        return image, targets


class DetectionCropToRatio(_BaseTransform):
    def __init__(self, max_ratio=1, min_ratio=1, mode="center", **kwargs):
        super().__init__(**kwargs)
        assert mode in ("center", "random")
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.mode = mode

    def joint_transform(self, image, label):
        """
        image: torch.Tensor (C, H, W)
        label: list[dict] - [{"bbox": [x, y, w, h], "cls": str}, ...]
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
                pad_h = random.randint(h - target_h)
                h_min, h_max = pad_h, pad_h + target_h
        elif ratio < self.min_ratio:
            # crop along width
            target_w = int(h / self.min_ratio)
            if self.mode == "center":
                pad_w = (w - target_w) // 2
                w_min, w_max = pad_w, pad_w + target_w
            elif self.mode == "random":
                pad_w = random.randint(w - target_w)
                w_min, w_max = pad_w, pad_w + target_w

        # crop image
        cropped_image = image[:, h_min:h_max, w_min:w_max]
        # move bbox(given in relative [x, y, w, h] coordinates)
        shifted_label = []
        for idx in range(len(label)):
            x1, y1, x2, y2 = xywh_to_x1y1x2y2(label[idx]["bbox"])
            # check if bbox is inside cropped image

            label[idx]["bbox"][0] += w_min / w
            label[idx]["bbox"][1] += h_min / h

        return cropped_image, shifted_label


class DetectionConstrainImageSize(object):
    # https://github.com/facebookresearch/maskrcnn-benchmark/blob/main/maskrcnn_benchmark/data/transforms/transforms.py
    def __init__(self, min_size, max_size):    
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
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

    def joint_transform(self, image, label):
        size = self.get_size(image.size)
        image = TF.resize(image, size)
        # target = target.resize(image.size)
        return image, label


class DetectionPadToRatio(_BaseTransform):
    # todo
    def joint_transform(self, image, label):
        """
        image: torch.Tensor (C, H, W)
        label: list[dict] - [{"bbox": [x, y, w, h], "cls": str}, ...]
        """
        pass


class DetectionHFlip(_BaseTransform):
    def __init__(self, prob=0.5):
        self.prob = prob

    def joint_transform(self, image, label):
        """
        image: torch.Tensor (C, H, W)
        label: list[dict] - [{"bbox": [x, y, w, h], "cls": str}, ...]
        """

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = TF.hflip(image)
            for idx in range(len(target)):
                target[idx]["bbox"][0] = 1. - target[idx]["bbox"][0]
        return image, target


class DetectionVFlip(_BaseTransform):
    def __init__(self, prob=0.5):
        self.prob = prob

    def joint_transform(self, image, label):
        """
        image: torch.Tensor (C, H, W)
        label: list[dict] - [{"bbox": [x, y, w, h], "cls": str}, ...]
        """

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = TF.vflip(image)
            for idx in range(len(target)):
                target[idx]["bbox"][1] = 1. - target[idx]["bbox"][1]
        return image, target
