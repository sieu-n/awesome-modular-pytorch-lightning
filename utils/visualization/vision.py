import cv2
from utils.bbox import unnormalize_bbox, xywh_to_x1y1x2y2

fontFont = cv2.FONT_HERSHEY_SIMPLEX
fontLineType = cv2.LINE_AA


def plot_image_classification(images, labels=None, label_map=None, **kwargs):
    """
    x: np.array(W, H, C)
        rgb image
    y (optional): int / str
    """
    image = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
    # x is cv2-style bgr image
    fontLocation = (10, 10)
    fontScale = 1
    fontColor = (255, 0, 0)
    fontThickness = 2
    if labels is not None:
        label = int(labels)
        # convert label to correct str id if specified.
        if label_map and label in label_map:
            label = label_map[label]
        image = cv2.putText(
            image,
            f"class: {str(label)}",
            fontLocation,
            fontFont,
            fontScale,
            fontColor,
            fontThickness,
            fontLineType,
        )
    # x is rgb image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def plot_object_detection(
    images, boxes=None, labels=None, label_map=None, is_xywh=True, **kwargs
):
    """
    x: np.array(W, H, C)
        rgb image
    y (optional): list[dict {"labels", "boxes": [x, y, w, h]}]
    """
    img = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
    # x is cv2-style bgr image
    gtColor = (0, 0, 255)
    # predColor = (0, 255, 0)
    fontScale = 0.5
    fontThickness = 1
    bboxThickness = 1
    if boxes is not None:
        img = cv2.putText(
            img,
            "- gt bbox",
            (0, 0),
            fontFont,
            fontScale,
            gtColor,
            fontThickness,
            fontLineType,
        )

        # plot gt bbox in red
        obj_classes, obj_bboxes = labels, boxes
        for obj_idx in range(len(obj_classes)):
            obj_class = int(obj_classes[obj_idx])
            obj_bbox = list(obj_bboxes[obj_idx])

            if label_map and obj_class in label_map:
                # convert label to correct str id if specified.
                obj_class = label_map[obj_class]
            if type(obj_bbox[0]) == float:
                w, h = img.shape[1], img.shape[0]
                obj_bbox = unnormalize_bbox(obj_bbox, w, h)
            if is_xywh:
                x1, y1, x2, y2 = xywh_to_x1y1x2y2(obj_bbox)
            else:
                x1, y1, x2, y2 = obj_bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # plot bbox
            img = cv2.rectangle(img, (x1, y1), (x2, y2), gtColor, bboxThickness)

            img = cv2.putText(
                img,
                str(obj_class),
                (x1, y1 - 5),
                fontFont,
                fontScale,
                gtColor,
                fontThickness,
                fontLineType,
            )
    # x is rgb image
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return x
