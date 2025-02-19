import cv2
import numpy as np
from .blend import cubic_blend, cubic_blend_2d


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
    mat,
    radius=1,
    corners=None):
    """
        source * matして得られるreferenceのピクセルで置換
    """
    ref_h, ref_w, *_ = reference.shape
    src_h, src_w, *_ = source.shape

    def create(i, j):
        pos = np.einsum('ij, jkl->ikl',
            mat, np.array([j, i, np.ones(shape=j.shape)]))
        pos = (pos[0:2] // pos[2]).astype(np.int16)

        if corners is not None:
            pos_min_w = (0 <= pos[0]) & (pos[0] <= radius)
            pos_min_h = (0 <= pos[1]) & (pos[1] <= radius)
            pos_max_w = (ref_w-1-radius <= pos[0]) & (pos[0] <= ref_w-1)
            pos_max_h = (ref_h-1-radius <= pos[1]) & (pos[1] <= ref_h-1)

            # sourceの四つ角位置を出力
            minh_minw_id = np.where(pos_min_w & pos_min_h) 
            minh_maxw_id = np.where(pos_max_w & pos_min_h)
            maxh_maxw_id = np.where(pos_max_w & pos_max_h)
            maxh_minw_id = np.where(pos_min_w & pos_max_h)

            if  minh_minw_id[0].size > 0 and minh_minw_id[1].size > 0 and\
                minh_maxw_id[0].size > 0 and minh_maxw_id[1].size > 0 and\
                maxh_maxw_id[0].size > 0 and maxh_maxw_id[1].size > 0 and\
                maxh_minw_id[0].size > 0 and maxh_minw_id[1].size > 0:
                
                print(
                    np.min(np.sqrt(np.sum((np.array(minh_minw_id).T  - corners)**2, axis=1))) +\
                    np.min(np.sqrt(np.sum((np.array(minh_maxw_id).T  - corners)**2, axis=1))) +\
                    np.min(np.sqrt(np.sum((np.array(minh_maxw_id).T  - corners)**2, axis=1))) +\
                    np.min(np.sqrt(np.sum((np.array(maxh_minw_id).T  - corners)**2, axis=1)))
                )
                print(minh_minw_id)
                print(minh_maxw_id)
                print(minh_maxw_id)
                print(maxh_minw_id)
                print(corners)

        u = np.clip(pos[0], 0, ref_w-1)
        v = np.clip(pos[1], 0, ref_h-1)
        return np.where((
            (pos[0] >= 0) & (pos[0] < ref_w) &\
            (pos[1] >= 0) & (pos[1] < ref_h))[:,:,None],
            reference[v, u],
            source)

    return np.fromfunction(create, shape=(src_h, src_w))


def warp_only(
    reference,
    source,
    mat,
    offset_h=0,
    offset_w=0):
    ref_h, ref_w, *_ = reference.shape
    src_h, src_w, *_ = source.shape

    def create(i, j):
        pos = np.einsum('ij, jkl->ikl',
            mat, np.array([j, i, np.ones(shape=j.shape)]))
        pos = (pos[0:2] // pos[2]).astype(np.int16)
        u = np.clip(pos[0]+offset_w, 0, ref_w-1)
        v = np.clip(pos[1]+offset_h, 0, ref_h-1)
        return np.where((
            (pos[0] >= 0) & (pos[0] < ref_w) &\
            (pos[1] >= 0) & (pos[1] < ref_h))[:,:,None],
            reference[v, u],
            0)

    return np.fromfunction(create, shape=(src_h, src_w))

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


def warp2(    
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
    src_points = np.array([
        to_bottom_left,
        to_bottom_right,
        to_top_right,
        to_top_left])

    dst_points = np.array([
        from_bottom_left, 
        from_bottom_right,
        from_top_right,
        from_top_left])

    src_x = src_points[:,1]
    src_y = src_points[:,0]
    dst_x = dst_points[:,1]
    dst_y = dst_points[:,0]

    shp = src_x.shape
    zero = lambda : np.zeros(shp)
    one = lambda : np.ones(shp)

    fst = np.array([zero(), zero(), zero(), src_x, src_y, one(), -dst_y * src_x, -dst_y * src_y])
    scd = np.array([src_x, src_y, one(), zero(), zero(), zero(), -dst_x * src_x, -dst_x * src_y])

    A = np.insert(fst, range(fst.shape[1]), scd, axis=1).T

    h = np.linalg.pinv(A) @ np.insert(dst_y,  range(dst_y.shape[0]), dst_x, axis=0)
    h = h.squeeze()
    h = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1]
    ])

    return warp_only(
        image,
        np.zeros(shape=(return_height, return_width)),
        h)


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

    # 任意の点から各辺への距離(d)とその垂線の足が辺のどのあたりに位置しているかの割合(t)を返す関数を作成
    def td(f, t):
        a = np.array([f, t])
        # fとtの差分=fを基準としたtへのベクトル
        dif = np.diff(a, axis=0)[0]
        det = np.linalg.det(a)
        sq = dif @ dif
        f = f[:,None,None]
        def ret(x):
            # 直線を表す式：dif[1] * x[0] - dif[0] * x[1] - det=0
            up = (dif[1] * x[0] - dif[0] * x[1] - det)
            r = -up / sq
            # fを基準とした垂線の足へのベクトル
            h_dif = f - np.array([r * dif[1] + x[0], -r * dif[0] + x[1]])
            # ベクトルの大きさの比と点と直線の距離を返戻
            return np.sqrt(np.sum(h_dif**2, axis=0) / sq), (np.abs(up) / np.sqrt(sq))
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

    h, w, *_ = image.shape

    # 関数の本体：i,jはnp.meshgridで与えられる
    def create(i, j):
        nonlocal to_bottom_left, to_bottom_right, to_top_left, to_top_right
        pos = np.array([i, j])

        # ピクセルが変換先の領域内に含まれているか判定するために外積を求める
        crs_bl = np.cross(
            pos - to_bottom_left[:,None, None],
            to_bottom_right - to_bottom_left,
            axis=0)
        crs_br = np.cross(
            pos - to_bottom_right[:,None, None],
            to_top_right - to_bottom_right, 
            axis=0)
        crs_tr = np.cross(
            pos - to_top_right[:,None, None],
            to_top_left - to_top_right, 
            axis=0)
        crs_tl = np.cross(
            pos - to_top_left[:,None, None],
            to_bottom_left - to_top_left, 
            axis=0)

        # posから各頂点への外積が負なら内部
        mask = np.where(
            (crs_bl < 0) & (crs_br < 0) & (crs_tr < 0) & (crs_tl < 0), 
            1, 0)
        t_bottom, d_bottom = td_bottom(pos)
        t_top, d_top = td_top(pos)
        t_left, d_left = td_left(pos)
        t_right, d_right = td_right(pos)

        t_vert = d_bottom / (d_bottom + d_top)
        t_hori = d_left / (d_left + d_right)

        v = (i\
            +((1 - t_bottom) * move_bottom_left[0] + t_bottom * move_bottom_right[0]) * (1 - t_vert)\
            + ((1 - t_top) * move_top_left[0] + t_top * move_top_right[0]) * t_vert)

        u = (j\
            + ((1 - t_left) * move_bottom_left[1] + t_left * move_top_left[1]) * (1 - t_hori)\
            + ((1 - t_right) * move_bottom_right[1] + t_right * move_top_right[1]) * t_hori)
        
        # バイリニア補完するために隣接ピクセルの影響度を計算
        v_int = v.astype(np.int16)
        u_int = u.astype(np.int16)
        bottom = np.where((v_int < 0) | (v_int > h-2), h-2, v_int)
        left = np.where((u_int < 0) | (u_int > w-2), w-2, u_int)
        top = bottom+1
        right = left+1
        v_ratio = (v - v_int)[:,:,None]
        u_ratio = (u - u_int)[:,:,None]

        return np.where((
                # 内部判定
                (mask == 0) |\
                # 辺の外側に出ていないか判定
                (((t_bottom < 0) | (t_bottom > 1)) &\
                ((t_top < 0) | (t_top > 1)) &\
                ((t_left < 0) | (t_left > 1)) &\
                ((t_right < 0) | (t_right > 1)) &\
                ((t_vert < 0) | (t_vert > 1)) &\
                ((t_hori < 0) | (t_hori > 1))) |\
                # 存在するピクセルを参照しているか判定
                ((v_int < 0) | (v_int > h-2) | (u_int < 0) | (u_int > w-2))
            )[:,:,None],
            0, 
            # バイリニア補完で色を決定
            cubic_blend(
                0, 
                image[bottom, left],
                image[top, left],
                image[bottom, right],
                image[top, right],
                v_ratio, u_ratio).astype(np.uint8))

    return np.fromfunction(create, shape=(return_height, return_width))

if __name__ == "__main__":
    fac = 10
    from operation import cross, diff
    import cv2
    img1 = np.ones(shape=(128, 128, 3), dtype=np.uint8) * 255
    bl = np.array((1, 2))*fac
    br = np.array((0, 5))*fac
    tr = np.array((6, 10))*fac
    tl = np.array((6, 0))*fac


    vec01 = diff(*br, *bl)
    vec12 = diff(*tr, *br)
    vec23 = diff(*tl, *tr)
    vec30 = diff(*bl, *tl)
    clip = np.zeros(shape=(15*fac, 16*fac, 3), dtype=np.uint8)
    rep = cv2.imread(r"C:\Users\Lab\Pictures\Camera Roll\WIN_20200710_01_05_18_Pro.jpg")
    for i in range(15*fac):
        for j in range(16*fac):
            if cross(*vec01, *diff(i, j, *bl)) < 0 and \
                cross(*vec12, *diff(i, j, *br)) < 0 and \
                cross(*vec23, *diff(i, j, *tr)) < 0 and \
                cross(*vec30, *diff(i, j, *tl)) < 0:
                clip[i, j] = rep[i, j]

    to_bl = bl+(np.array([1,2])+5)*fac
    to_br = br+(np.array([1,1])+5)*fac
    to_tr = tr+(np.array([1,4])+5)*fac
    to_tl = tl+(np.array([1,2])+5)*fac
    ret = warp2(
        clip,
        bl,
        br,
        tr,
        tl,
        to_bl,
        to_br,
        to_tr,
        to_tl,
        26*fac, 26*fac)
    
    import time
    sum = 0
    for i in range(10000):
        prev = time.time()
        warp2(
            clip,
            bl,
            br,
            tr,
            tl,
            to_bl,
            to_br,
            to_tr,
            to_tl,
            256, 256)
        sum += time.time() - prev
    print(sum / 10000)

    sum = 0
    for i in range(10000):
        prev = time.time()
        warp(
            clip,
            bl,
            br,
            tr,
            tl,
            to_bl,
            to_br,
            to_tr,
            to_tl,
            256, 256)
        sum += time.time() - prev
    print(sum / 10000)

    ret[to_bl[0], to_bl[1]] = [0,0,255]
    ret[to_br[0], to_br[1]] = [0,255,0]
    ret[to_tr[0], to_tr[1]] = [255,0,255]
    ret[to_tl[0], to_tl[1]] = [255,255,0]

    cv2.imwrite("start.png", img1)
    cv2.imwrite("from.png", clip)
    cv2.imwrite("to2.png", ret)
