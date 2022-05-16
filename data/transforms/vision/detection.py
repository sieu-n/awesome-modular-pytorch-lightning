import torchvision.transforms.functional as TF
from data.transforms.common import _BaseTransform
from utils.bbox import pixel_bbox_to_relative, x1y1x2y2_to_xywh
import random


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
        self.label2code = {name: idx for name, idx in enumerate(self.const_cfg["label_map"])}

    def joint_transform(self, image, label):
        img_w, img_h = image.size(2), image.size(1)
        label = label["annotation"]["object"]
        # returns [{"bbox": [x1, y1, x2, y2], "cls": str}, ...]
        targets = []
        for obj_label in label:
            bbox = obj_label["bndbox"]
            bbox_xywh = x1y1x2y2_to_xywh([bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]])
            targets.append({
                "bbox": pixel_bbox_to_relative(bbox_xywh, img_w, img_h),
                "class": self.label2code[obj_label["name"]],
            })
        return image, targets


class DetectionCropToRatio(_BaseTransform):
    def __init__(self, max_ratio=1, min_ratio=1, mode="center", **kwargs):
        super().__init__(**kwargs)
        assert mode in ("center", "random")
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.mode = mode

    def joint_transform(self, image, label):
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

        return {"w_min": w_min, "w_max": w_max, "h_min": h_min, "h_max": h_max}, {"w_translate"}


class DetectionPadToRatio(_BaseTransform):
    # todo
    pass

class DetectionHFlip(_BaseTransform):
    # todo
    pass

class DetectionVFlip(_BaseTransform):
    # todo
    pass

