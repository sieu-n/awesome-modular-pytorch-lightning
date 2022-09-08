import torch
from data.transforms import _BaseTransform
from utils.bbox import normalize_bbox, x1y1x2y2_to_xywh


class DetectionVOCLabelTransform(_BaseTransform):
    def __init__(self, **kwargs):
        """
        Recieves the naive `VOC2012` torchvision dataset, which contains the annotations from the `.xml` file.
        Processes and returns the class and bbox in the following format:

        x = PIL.Image
        y = {"boxes": torch.Tensor(num_obj, 4), "labels": torch.Tensor(num_obj)}

        Each bbox coordinates are given in (x, y, w, h) format. The numbers are normalized to (0, 1) range by dividing
        them with the width and height of the image. This is similar to the YOLO bounding box format.
        """
        super().__init__(**kwargs)
        self.label2code = {
            name: idx for idx, name in enumerate(self.const_cfg["label_map"])
        }

    def __call__(self, x, y):
        boxes, labels = self.transform(y, x.size[0], x.size[1])
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

        # returns {"boxes": torch.Tensor(num_obj, 4), "labels": torch.Tensor(num_obj)}
        return torch.tensor(targets["boxes"]), torch.tensor(targets["labels"])
