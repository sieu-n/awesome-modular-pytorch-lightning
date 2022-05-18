import cv2
from utils.bbox import pixel_bbox_to_absolute, xywh_to_x1y1x2y2

fontFont = cv2.FONT_HERSHEY_SIMPLEX
fontLineType = cv2.LINE_AA


def plot_image_classification(x, pred=None, y=None, label_map=None):
    """
    x: np.array(W, H, C)
        rgb image
    y (optional): int / str
    """
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    # x is cv2-style bgr image
    fontLocation = (10, 10)
    fontScale = 1
    fontColor = (255, 0, 0)
    fontThickness = 2
    if y:
        # convert label to correct str id if specified.
        if label_map:
            y = label_map[y]
        x = cv2.putText(
            x,
            f"class: {str(y)}",
            fontLocation,
            fontFont,
            fontScale,
            fontColor,
            fontThickness,
            fontLineType,
        )
    # x is rgb image
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x


def plot_object_detection(x, pred=None, y=None, label_map=None):
    # todo: fix bug where bbox is not displayed.
    """
    x: np.array(W, H, C)
        rgb image
    y (optional): list[dict {"class", "bbox": [x, y, w, h]}]
    """
    img = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    # x is cv2-style bgr image
    gtColor = (0, 0, 255)
    predColor = (0, 255, 0)
    fontScale = 0.5
    fontThickness = 2
    bboxThickness = 1
    if y:
        # plot gt bbox in red
        for obj in y:
            obj_class, obj_bbox = obj["class"], obj["bbox"]
            if label_map:
                # convert label to correct str id if specified.
                obj_class = label_map[obj_class]
            if type(obj_bbox[0]) == float:
                w, h = img.shape[1], img.shape[0]
                y_bbox = pixel_bbox_to_absolute(obj_bbox, w, h)
            # plot bbox

            x1, y1, x2, y2 = xywh_to_x1y1x2y2(y_bbox)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
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
