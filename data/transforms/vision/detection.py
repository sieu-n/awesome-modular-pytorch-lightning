import torchvision.transforms.functional as TF
from data.transforms.common import _BaseTransform
from utils.bbox import pixel_bbox_to_relative

class DetectionVOCLabelTransform(_BaseTransform):
    def __init__(self, label_map):
        self.label_map = label_map

    def set_transform(self, image, label):
        img_w, img_h = image.size(2), image.size(1)
        return {}, {"img_w": img_w, "img_h": img_h}
    def label_transform(self, label, img_w, img_h):
        label = label["annotation"]["object"]
        # returns [{"bbox": [x1, y1, x2, y2], "cls": str}, ...]
        targets = []
        for l in label:
            bbox = l["bbox"]
            targets.append({
                "bbox": pixel_bbox_to_relative([bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]], img_w, img_h),
                "class": self.label_map[l["name"]],
            })
        return targets


class DetectionCropToRatio(_BaseTransform):
    def __init__(self, max_ratio=1, min_ratio=1, mode="center"):
        assert mode in ("center", "random")
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.mode = mode

    def set_transform(self, image, label):
        h, w = image.size(1), image.size(2)
        ratio = w / h
        w_min, w_max = 0, w
        h_min, h_max = 0, h
        if ratio < self.min_ratio:
            # crop height
            target_h = int(w / self.min_ratio)
            if self.mode == "center":
                pad_h = (h - target_h) // 2
                h_min, h_max = pad_h, pad_h + target_h
            elif self.mode == 
        elif ratio > self.max_ratio:
            # crop width
        return {"w_min": w_min, "w_max": w_max, "h_min": h_min, "h_max": h_max}, {"w_translate"}

    def input_transform(self, image):
        return image

    def label_transform(self, label):
        return label


class DetectionPadToRatio(_BaseTransform):
