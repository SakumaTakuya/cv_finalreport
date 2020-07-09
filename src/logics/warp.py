import numpy as np
from blend import cubic_blend, cubic_blend_2d


def warp_image_liner(
    image,
    from_left_bottom_y, 
    from_left_bottom_x, 
    from_right_bottom_y,
    from_right_bottom_x,
    from_right_top_y, 
    from_right_top_x,
    from_left_top_y, 
    from_left_top_x, 
    height,
    width):
    move_left_bottom_y = from_left_bottom_y - 0
    move_left_bottom_x = from_left_bottom_x - 0
    move_right_bottom_y = from_right_bottom_y - 0
    move_right_bottom_x = from_right_bottom_x - width
    move_left_top_y = from_left_top_y - height
    move_left_top_x = from_left_top_x - 0
    move_right_top_y = from_right_top_y - height
    move_right_top_x = from_right_top_x - width
    _, _, color_channel = image.shape
    ret = np.zeros(shape=(height, width, color_channel), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            # u, v = cubic_blend_2d(
            #         i, j, 
            #         move_left_bottom_x, move_left_bottom_y,
            #         move_right_bottom_x, move_right_bottom_y,
            #         move_left_top_x, move_left_top_y,
            #         move_right_top_x, move_right_top_y,
            #         i / height, j / width)
            v, u = cubic_blend_2d(
                    i, j, 
                    move_left_bottom_y, move_left_bottom_x,
                    move_right_bottom_y, move_right_bottom_x,
                    move_left_top_y, move_left_top_x,
                    move_right_top_y, move_right_top_x,
                    i / height, j / width)
            left = int(u)
            right = left+1
            bottom = int(v)
            top = bottom+1
            u_ratio = u - left
            v_ratio = v - bottom
            
            ret[i, j] = cubic_blend(
                            0, 
                            image[bottom, left],
                            image[bottom, right],
                            image[top, left],
                            image[top, right],
                            v_ratio, u_ratio)

    return ret


if __name__ == "__main__":
    siz = 160
    from operation import cross, diff
    img1 = np.ones(shape=(siz, siz, 1), dtype=np.uint8) * 255
    bl = (10, 20)
    br = (60, 0)
    tr = (60, 100)
    tl = (0, 50)

    img1[bl] = 0
    img1[br] = 0
    img1[tr] = 0
    img1[tl] = 0

    vec01 = diff(*br, *bl)
    vec12 = diff(*tr, *br)
    vec23 = diff(*tl, *tr)
    vec30 = diff(*bl, *tl)
    clip = np.zeros(shape=(siz, siz, 1), dtype=np.uint8)
    for i in range(siz):
        for j in range(siz):
            if cross(*vec01, *diff(i, j, *bl)) > 0 and \
                cross(*vec12, *diff(i, j, *br)) > 0 and \
                cross(*vec23, *diff(i, j, *tr)) > 0 and \
                cross(*vec30, *diff(i, j, *tl)) > 0:
                clip[i, j] = img1[i, j] // np.random.randint(1, 8, dtype=np.uint8)

    ret = warp_image_liner(
        clip,
        *bl,
        *br,
        *tr,
        *tl,
        128, 256)

    import cv2
    cv2.imwrite("start.png", img1)
    cv2.imwrite("from.png", clip)
    cv2.imwrite("to.png", ret)
