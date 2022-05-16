import cv2
from utils.bbox import pixel_bbox_to_absolute

fontFont = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontLineType = cv2.LINE_AA


def plot_image_classification(x, pred=None, y=None, label_map=None):
    """
    x: np.array(W, H, C)
        rgb image
    y (optional): int / str
    """
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    # x is cv2-style bgr image
    fontLocation = (50, 50)
    fontColor = (255, 0, 0)
    fontThickness = 2
    if y:
        # convert label to correct str id if specified.
        if label_map:
            y = label_map[y]
        x = cv2.putText(x, f"class: {str(y)}", fontLocation, fontFont,
                        fontScale, fontColor, fontThickness, fontLineType)
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
    fontColor = (255, 0, 0)
    fontThickness = 2
    bboxThickness = 1
    if y:
        y_class, y_bbox = y["class"], y["bbox"]
        # convert label to correct str id if specified.
        if label_map:
            y_class = label_map[y_class]
        if type(y_bbox[0]) == float:
            w, h = img.shape[1], img.shape[0]
            y_bbox = pixel_bbox_to_absolute(y_bbox, w, h)
        # plot bbox

        x1, y1, x2, y2 = y_bbox
        img = cv2.rectangle(img, x1, y1, x2, y2, fontColor, bboxThickness)
        img = cv2.putText(img, y_class, (x1, y1 - 5), fontFont,
                          fontScale, fontColor, fontThickness, fontLineType)
    # x is rgb image
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return x
