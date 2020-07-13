"""
    sift:
        https://www.slideshare.net/la_flance/sift-20150311
        https://www.slideshare.net/hironobufujiyoshi/miru2013sift
    scale space for sift:
        https://qiita.com/JunyaKaneko/items/e3b5e4d4c249847e9b5f
"""
import numpy as np
from scipy import ndimage


def luminance(x):
    return x[:,:,0] + x[:,:,1] + x[:,:,2]


def normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord=order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return v/l2


def gauss(x, a=1.0, mu=0.0, sigma=1.0):
    return a * np.exp(-(x - mu)**2 / (2.0*sigma**2))


def gaussian_convolution(image, sigma, truncate=4.0):
    h, w, *_ = image.shape
    radius = min(int(truncate * sigma + 0.5), min(h, w)) // 2
    filter = normalize(gauss(np.arange(-radius, radius, 1), sigma=sigma))
    filtered = np.apply_along_axis(lambda m: np.convolve(m, filter, mode='same'), axis=0, arr=image)
    filtered = np.apply_along_axis(lambda m: np.convolve(m, filter, mode='same'), axis=1, arr=filtered)
    return filtered


def scale_space(image, sigma0=1.6, s=4):
    def func(x, y):
        return ndimage.filters.gaussian_filter(image[::y, ::y], x)
    
    h, w, *_ = image.shape
    k = 2**(1/s)
    std = sigma0 * s
    octove_num = int(np.log2(min(h, w)))
    
    octove_std = np.power(np.power(k, 2), np.arange(0, octove_num))
    picture_std = np.power(k, np.arange(s))
    stds = np.outer(octove_std, picture_std) * std
    
    s_h, s_w = stds.shape
    resizes = np.tile(2**np.arange(s_h), (s_w, 1)).T
    return np.frompyfunc(func, 2, 1)(stds, resizes), stds


def difference_of_gaussian(scale_space_image):
    return np.apply_along_axis(np.diff, axis=1, arr=scale_space_image)


def locate_extreme_value(dogs):
    def func(arr):
        # 同じオクターブでは画像サイズが同じ
        arr = np.array([a for a in arr])
        t, y, x, *_ = arr.shape

        kps = []
        for y_id in range(1, y-1):
            for x_id in range(1, x-1):
                for t_id in range(1, t-1):
                    val = arr[t_id,y_id,x_id]
                    sub = arr[t_id-1:t_id+2,y_id-1:y_id+2,x_id-1:x_id+2]

                    if val == np.min(sub) or val != np.max(sub):
                        continue

                    dx = (sub[1,1,2] - sub[1,1,0]) * 0.5 / 255
                    dy = (sub[1,2,1] - sub[1,0,1]) * 0.5 / 255
                    dt = (sub[2,1,1] - sub[0,1,1]) * 0.5 / 255
                    part = np.matrix([
                        [dx], 
                        [dy], 
                        [dt]])

                    dxx = (sub[1,1,2] + sub[1,1,0] - 2 * val) / 255
                    dyy = (sub[1,2,1] + sub[1,0,1] - 2 * val) / 255
                    dtt = (sub[2,1,1] + sub[0,1,1] - 2 * val) / 255
                    
                    dxy = (sub[1,2,2] - sub[1,2,0] - sub[1,0,2] + sub[1,0,0]) * 0.25 / 255
                    dxt = (sub[2,1,2] - sub[2,1,0] - sub[0,1,2] + sub[0,1,0]) * 0.25 / 255
                    dyt = (sub[2,2,1] - sub[2,0,1] - sub[0,2,1] + sub[0,0,1]) * 0.25 / 255

                    hess = np.array([
                        [dxx, dxy, dxt],
                        [dxy, dyy, dyt],
                        [dxt, dyt, dtt]])

                    try:
                        x_hat = -np.linalg.inv(hess) @ part
                    except:
                        continue

                    p = np.abs(val + 0.5 * (part.T @ x_hat))
                    det_H2 = (dxx * dyy) - (dxy ** 2)
                    trace_H2 = (dxx + dyy) ** 2

                    if  p < 0.03 or det_H2 <= 0 or\
                        trace_H2 / det_H2 > 12 or\
                        np.count_nonzero(x_hat < 0.5) != 3:
                        continue

                    kps.append((x_id, y_id))
        return kps
    return [func(d) for d in dogs]


def find_keypoints(image, sigma0=1.6, s=4):
    image = image.astype(np.float64)
    scale, stds = scale_space(image, sigma0=sigma0, s=s)
    dogs = difference_of_gaussian(scale)
    return locate_extreme_value(dogs), list(stds)




if __name__ == "__main__":
    import cv2
    im_gray = cv2.imread(r"C:\Users\Lab\Documents\sakuma\class\computervision\cv_finalreport\reference.png", cv2.IMREAD_GRAYSCALE)
    im_gray = im_gray.astype(np.float64)
    scale, stds = scale_space(im_gray)
    for o, s in enumerate(scale):
        for i, im in enumerate(s):
            cv2.imwrite(f"test_res\\scale_oct{o}_{i}.png", im.astype(np.uint8))
    dogs = difference_of_gaussian(scale)
    for o, s in enumerate(dogs):
        for i, im in enumerate(s):
            cv2.imwrite(f"test_res\\dogs_oct{o}_{i}.png", im.astype(np.uint8))

    kps = locate_extreme_value(dogs)
    print(kps)


    sift = cv2.xfeatures2d.SIFT_create()
    img = cv2.imread(r"C:\Users\Lab\Documents\sakuma\class\computervision\cv_finalreport\reference.png")
    kp, des = sift.detectAndCompute(img, None)
    print(len(kp), len(des))
