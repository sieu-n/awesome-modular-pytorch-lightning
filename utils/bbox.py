def pixel_bbox_to_relative(coord, img_w, img_h):
    # either [x1 y1 x2 y2] or [x y w h]
    return [coord[0] / img_w, coord[1] / img_h, coord[2] / img_w, coord[3] / img_h]


def pixel_bbox_to_absolute(coord, img_w, img_h):
    # either [x1 y1 x2 y2] or [x y w h]
    return [coord[0] * img_w, coord[1] * img_h, coord[2] * img_w, coord[3] * img_h]


def xywh_to_x1y1x2y2(xywh):
    return [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]


def x1y1x2y2_to_xywh(x1y1x2y2):
    return [(x1y1x2y2[0] + x1y1x2y2[2]) / 2,
            (x1y1x2y2[1] + x1y1x2y2[3]) / 2,
            (x1y1x2y2[2] - x1y1x2y2[0]) / 2,
            (x1y1x2y2[3] - x1y1x2y2[1]) / 2]
