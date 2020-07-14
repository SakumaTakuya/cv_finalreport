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
    # numpyならmeshgridで書き換えられそう
    for i in range(height):
        for j in range(width):
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
                            image[top, left],
                            image[bottom, right],
                            image[top, right],
                            v_ratio, u_ratio)

    return ret


def replace_image(
    reference,
    source,
    mat):
    """
        source * matして得られるreferenceのピクセルで置換
    """
    ref_h, ref_w, color_channel = reference.shape
    src_h, src_w, color_channel = source.shape

    # harris = cv2.cornerHarris(source, 2, 3, 0.04)
    # inv = np.linalg.inv(mat)
    # c_00 = (inv @ np.array([    0,    0,1])).astype(np.uint8)
    # c_01 = (inv @ np.array([    0,ref_h,1])).astype(np.uint8)
    # c_10 = (inv @ np.array([ref_w,    0,1])).astype(np.uint8)
    # c_11 = (inv @ np.array([ref_w,ref_h,1])).astype(np.uint8)

    # radius = 10
    # max_00 = np.argmax(harris[c_00[0]-radius:c_00[0]+radius, c_00[1]-radius:c_00[1]+radius])
    # min_00 = np.argmin(harris[c_00[0]-radius:c_00[0]+radius, c_00[1]-radius:c_00[1]+radius])
    # max_01 = np.argmax(harris[c_01[0]-radius:c_01[0]+radius, c_01[1]-radius:c_01[1]+radius])
    # min_01 = np.argmin(harris[c_01[0]-radius:c_01[0]+radius, c_01[1]-radius:c_01[1]+radius])
    # max_10 = np.argmax(harris[c_10[0]-radius:c_10[0]+radius, c_10[1]-radius:c_10[1]+radius])
    # min_10 = np.argmin(harris[c_10[0]-radius:c_10[0]+radius, c_10[1]-radius:c_10[1]+radius])
    # max_11 = np.argmax(harris[c_11[0]-radius:c_11[0]+radius, c_11[1]-radius:c_11[1]+radius])
    # min_11 = np.argmin(harris[c_11[0]-radius:c_11[0]+radius, c_11[1]-radius:c_11[1]+radius])
    
    # thr = 0.01*harris.max()
    # if harris[max_00] > thr:



    ret = np.zeros(shape=source.shape)
    ids = np.zeros(shape=(4,2))
    # meshgridで書き換えられそう
    for i in range(src_h):
        for j in range(src_w):
            pos = mat @ np.array([j, i, 1])

            # scale処理
            pos = pos // pos[2]
            if  pos[0] >= 0 and \
                pos[0] < ref_h and \
                pos[1] >= 0 and \
                pos[1] < ref_w:
                ret[i, j] = reference[pos[0], pos[1]]
            
            if pos[0] == 0 and pos[1] == 0:
                ids[0] = np.array([j, i])
            if pos[0] == ref_h and pos[1] == 0:
                ids[1] = np.array([j, i])
            if pos[0] == 0 and pos[1] == ref_w:
                ids[2] = np.array([j, i])
            if pos[0] == ref_h and pos[1] == ref_w:
                ids[3] = np.array([j, i])

    return ret




def warp_image(
    image,
    from_bottom_left,
    from_bottom_right,
    from_top_right,
    from_top_left,
    to_bottom_left,
    to_bottom_right,
    to_top_right,
    to_top_left,
    return_height,
    return_width):
    rect = np.array([to_bottom_left,to_bottom_right,to_top_right,to_top_left])

    top_right = np.max(rect, axis=0) - np.min(rect, axis=0)
    bottom_left = np.array([0, 0])
    top_left = np.array([top_right[0], 0])
    bottom_right = np.array([0, top_right[1]])

    # 変換先の最大高さ・幅
    height = top_right[0]
    width = top_right[1]

    # 各頂点の移動量を計算
    from_move_bottom_left = from_bottom_left - bottom_left
    from_move_bottom_right = from_bottom_right - bottom_right
    from_move_top_left = from_top_left - top_left
    from_move_top_right = from_top_right - top_right

    to_move_bottom_left = to_bottom_left - bottom_left
    to_move_bottom_right = to_bottom_right - bottom_right
    to_move_top_left = to_top_left - top_left
    to_move_top_right = to_top_right - top_right
    
    _, _, color_channel = image.shape
    far = np.sqrt(2)

    ret = np.zeros(shape=(return_height, return_width, color_channel), dtype=np.uint8)
    # numpyならmeshgridで書き換えられそう
    for i in range(height):
        for j in range(width):
            from_v, from_u = cubic_blend_2d(
                    i, j, 
                    *from_move_bottom_left,
                    *from_move_bottom_right ,
                    *from_move_top_left,
                    *from_move_top_right,
                    i / height, j / width)
            to_v, to_u = cubic_blend_2d(
                    i, j, 
                    *to_move_bottom_left,
                    *to_move_bottom_right ,
                    *to_move_top_left,
                    *to_move_top_right,
                    i / height, j / width)
            from_left = int(from_u)
            from_right = from_left+1
            from_bottom = int(from_v)
            from_top = from_bottom+1
            from_u_ratio = from_u - from_left
            from_v_ratio = from_v - from_bottom
            
            color = cubic_blend(
                            0, 
                            image[from_bottom, from_left],
                            image[from_top, from_left],
                            image[from_bottom, from_right],
                            image[from_top, from_right],
                            from_v_ratio, from_u_ratio)

            # to_left = int(to_u)
            # to_right = to_left+1
            # to_bottom = int(to_v)
            # to_top = to_bottom+1
            # to_u_ratio = to_u - to_left
            # to_v_ratio = to_v - to_bottom

            # r_u = to_u_ratio ** 2
            # r_1_u = (1- to_u_ratio) ** 2
            # r_v = to_v_ratio ** 2
            # r_1_v = (1- to_v_ratio) ** 2

            # ret[to_bottom, to_left] += np.uint8(color * (1-np.sqrt(r_u+r_v) / far) / 4) 
            # ret[to_bottom, to_right] += np.uint8(color * (1-np.sqrt(r_1_u+r_v) / far) / 4)
            # ret[to_top, to_right] += np.uint8(color * (1-np.sqrt(r_1_u+r_1_v) / far) / 4)
            # ret[to_top, to_left] += np.uint8(color * (1-np.sqrt(r_u+r_1_v) / far) / 4)

            to_u = round(to_u)
            to_v = round(to_v)
            ret[to_v, to_u] = color
    return ret


def warp(
    image,
    from_bottom_left,
    from_bottom_right,
    from_top_right,
    from_top_left,
    to_bottom_left,
    to_bottom_right,
    to_top_right,
    to_top_left,
    return_height,
    return_width):

    # 任意の点から各辺への距離(d)とその垂線の足が辺のどのあたりに位置しているかの割合(t)
    def td(f, t):
        a = np.array([f, t])
        dif = np.diff(a, axis=0)[0]
        det = np.linalg.det(a)
        sq = dif @ dif
        def ret(x):
            up = (dif[1] * x[0] - dif[0] * x[1] - det)
            r = -up / sq
            h_dif = f - np.array([r * dif[1] + x[0], -r * dif[0] + x[1]]) 
            return np.sqrt(h_dif @ h_dif / sq), np.abs(up) / np.sqrt(sq) 
        return ret

    td_bottom = td(to_bottom_left, to_bottom_right)
    td_top = td(to_top_left, to_top_right)
    td_left = td(to_bottom_left, to_top_left)
    td_right = td(to_bottom_right, to_top_right)

    # 各頂点の移動量を計算
    move_bottom_left = from_bottom_left - to_bottom_left
    move_bottom_right = from_bottom_right - to_bottom_right
    move_top_left = from_top_left - to_top_left
    move_top_right = from_top_right - to_top_right

    h, w, color_channel = image.shape
    ret = np.zeros(shape=(return_height, return_width, color_channel), dtype=np.uint8)
    for i in range(return_height):
        for j in range(return_width):
            pos = np.array([i, j])
            t_bottom, d_bottom = td_bottom(pos)
            t_top, d_top = td_top(pos)
            t_left, d_left = td_left(pos)
            t_right, d_right = td_right(pos)

            if  (t_bottom < 0 or t_bottom > 1) and \
                (t_top < 0 or t_top > 1) and \
                (t_left < 0 or t_left > 1) and \
                (t_right < 0 or t_right > 1):
                continue

            
            t_vert = d_bottom / (d_bottom + d_top)
            t_hori = d_left / (d_left + d_right)

            v = i\
                +((1 - t_bottom) * move_bottom_left[0] + t_bottom * move_bottom_right[0]) * (1 - t_vert)\
                + ((1 - t_top) * move_top_left[0] + t_top * move_top_right[0]) * t_vert

            u = j\
                + ((1 - t_left) * move_bottom_left[1] + t_left * move_top_left[1]) * (1 - t_hori)\
                + ((1 - t_right) * move_bottom_right[1] + t_right * move_top_right[1]) * t_hori

            bottom = int(v)
            left = int(u)
            if  bottom < 0 or bottom > w-2 or\
                left < 0 or left > h-2:
                continue

            top = bottom+1
            right = left+1
            u_ratio = u - left
            v_ratio = v - bottom

            ret[i, j] = cubic_blend(
                            0, 
                            image[bottom, left],
                            image[top, left],
                            image[bottom, right],
                            image[top, right],
                            v_ratio, u_ratio)

    return ret

if __name__ == "__main__":
    # siz = 160
    # from operation import cross, diff
    # img1 = np.ones(shape=(siz, siz, 1), dtype=np.uint8) * 255
    # bl = (10, 20)
    # br = (60, 0)
    # tr = (60, 100)
    # tl = (0, 50)

    # img1[bl] = 0
    # img1[br] = 0
    # img1[tr] = 0
    # img1[tl] = 0

    # vec01 = diff(*br, *bl)
    # vec12 = diff(*tr, *br)
    # vec23 = diff(*tl, *tr)
    # vec30 = diff(*bl, *tl)
    # clip = np.zeros(shape=(siz, siz, 1), dtype=np.uint8)
    # for i in range(siz):
    #     for j in range(siz):
    #         if cross(*vec01, *diff(i, j, *bl)) > 0 and \
    #             cross(*vec12, *diff(i, j, *br)) > 0 and \
    #             cross(*vec23, *diff(i, j, *tr)) > 0 and \
    #             cross(*vec30, *diff(i, j, *tl)) > 0:
    #             clip[i, j] = img1[i, j] // np.random.randint(1, 8, dtype=np.uint8)

    # ret = warp_image_liner(
    #     clip,
    #     *bl,
    #     *br,
    #     *tr,
    #     *tl,
    #     160, 160)

    # import cv2
    # cv2.imwrite("start.png", img1)
    # cv2.imwrite("from.png", clip)
    # cv2.imwrite("to.png", ret)

    siz = 160
    from operation import cross, diff
    import cv2
    img1 = np.ones(shape=(siz, siz, 3), dtype=np.uint8) * 255
    bl = np.array((10, 20))
    br = np.array((0, 50))
    tr = np.array((60, 100)) 
    tl = np.array((60, 0))

    vec01 = diff(*br, *bl)
    vec12 = diff(*tr, *br)
    vec23 = diff(*tl, *tr)
    vec30 = diff(*bl, *tl)
    clip = np.zeros(shape=(siz, siz, 3), dtype=np.uint8)
    rep = cv2.imread(r"C:\Users\Lab\Documents\sakuma\class\computervision\cv_finalreport\destination.png")
    for i in range(siz):
        for j in range(siz):
            if cross(*vec01, *diff(i, j, *bl)) < 0 and \
                cross(*vec12, *diff(i, j, *br)) < 0 and \
                cross(*vec23, *diff(i, j, *tr)) < 0 and \
                cross(*vec30, *diff(i, j, *tl)) < 0:
                clip[i, j] = rep[i, j]

    to_bl = bl+np.array([10,20])+50
    to_br = br+np.array([15,10])+50
    to_tr = tr+np.array([12,40])+50
    to_tl = tl+np.array([10,21])+50
    ret = warp(
        clip,
        bl,
        br,
        tr,
        tl,
        to_bl,
        to_br,
        to_tr,
        to_tl,
        260, 260)

    ret[to_bl[0], to_bl[1]] = [0,0,255]
    ret[to_br[0], to_br[1]] = [0,255,0]
    ret[to_tr[0], to_tr[1]] = [255,0,255]
    ret[to_tl[0], to_tl[1]] = [255,255,0]

    cv2.imwrite("start.png", img1)
    cv2.imwrite("from.png", clip)
    cv2.imwrite("to.png", ret)
