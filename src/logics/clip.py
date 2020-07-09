
import numpy as np
from .operation import cross, diff

def clip_image(image, points):
    vec01 = diff(*points[1], *points[0])
    vec12 = diff(*points[2], *points[1])
    vec23 = diff(*points[3], *points[2])
    vec30 = diff(*points[0], *points[3])

    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    siz_y = max_y-min_y+1
    siz_x = max_x-min_x+1

    clip = np.zeros(shape=(siz_y, siz_x, 3), dtype=np.uint8)
    for i in range(min_y, max_y+1):
        for j in range(min_x, max_x+1):
            if cross(*vec01, *diff(j, i, *points[0])) > 0 and \
                cross(*vec12, *diff(j, i, *points[1])) > 0 and \
                cross(*vec23, *diff(j, i, *points[2])) > 0 and \
                cross(*vec30, *diff(j, i, *points[3])) > 0:
                clip[i-min_y, j-min_x] = image[i, j]

    return clip