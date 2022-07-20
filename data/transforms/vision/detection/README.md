TODO
# Data and bounding box formats

In `.convert_format.py`, we provide utilities for converting between different formats.

## Default detection data structure

```Python
{
    "images": Tensor[C, W, H],
    "boxes":  Tensor[n, 4],
    "labels":  Tensor[n],
    "meta": {
        (optional)
    }
    ...
}
```
Inspired by [this](Tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
) tutorial by torchvision, we use the format described above to store object detection data.

By default, the custom transform function assumes that the object detection data is in the format above. This format is also compatible with the standard image-only transforms that apply transformations to the data in the `images` key.

## Bounding box formats

There are a few different ways to represent bounding boxes for object detection. For example, popular formats include:

```txt
Coco Format: [x_min, y_min, width, height]
Pascal_VOC Format: [x_min, y_min, x_max, y_max]
YOLO Format: [x_center, y_center, width, height], where values are relative to the image width and height, normalized to [0, 1].
```

- Unless specified, transforms implemented in this repository will assume the `YOLO` bounding box format. 
- MMDetection use the Pascal_VOC format by default.
- We provide utilities for converting between different formats in `.convert_format.py`.

## MMDetection
We integrate dataset and models from `mmdetection`. To use the features, we must be aware of different formats.

```Python
{
    "gt_bboxes": Tensor[n, 4],
    "gt_labels": Tensor[n],
    "img": Tensor[C, W, H],
    "img_metas": {
        "filename": str,
        "flip": bool,
        "img_norm_cfg": {
            "mean": np.array[3],
            "std": np.array[3],,
            "to_rgb": bool,
        },
        "img_shape": tuple,
        "ori_filename": str,
        'ori_shape': tuple,
        'pad_shape': tuple,
        'scale_factor': np.array,
    }
}
```
- The validation subset should only have the `img` and `img_metas` fields.
- The example above shows `img_metas` of PascalVOC dataset. The field might difer based on the type and configuration of dataset. 
  - When converting custom datasets for training using MMDetection models, make sure that the `meta` field contains the keys required for training. On how to make custom datasets for MMDetection, refer to [this](https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb) tutorial.
  - Some models from mmdetection could require that more keys must be specified.
- Bounding boxes are in `(x1, y1, x2, y2)` order, where each value represents absolute pixel location of bounding boxes.

## Raw dataset formats

### VOC

# Converting between formats

- `Pytorchbbox2YOLO`: YOLO bbox -> VOC bbox
- `YOLObbox2Pytorch`: VOC bbox -> YOLO bbox
- `MMdetDataset2Torchvision`: MMdetection dataset -> torchvision dataset