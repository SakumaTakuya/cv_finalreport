import numpy as np
from scipy import ndimage


def delta(image):
    iy = np.diff(image, n=1, axis=0, prepend=0)
    ix = np.diff(image, n=1, axis=1, prepend=0)
    iyy = iy**2
    ixx = ix**2
    iyx = iy*ix
    import cv2
    cv2.imwrite("iy.png", iy*20)
    cv2.imwrite("ix.png", ix*20)
    cv2.imwrite("iyy.png", iyy)
    cv2.imwrite("ixx.png", ixx)
    cv2.imwrite("iyx.png", iyx)
    return np.array([
        [iyy, iyx],
        [iyx, ixx]
    ])


def normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord=order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return v/l2


def gauss(x, a=1.0, mu=0.0, sigma=1.0):
    return a * np.exp(-(x - mu)**2 / (2.0*sigma**2))


def gaussian_convolution(image, sigma, truncate=4.0, radius=None, y_axis=0, x_axis=1):
    h, w, *_ = image.shape
    radius = radius or min(int(truncate * sigma + 0.5), min(h, w)) // 2
    filt = gauss(np.arange(-radius, radius+1, 1), sigma=sigma)
    filt /= sum(filt)
    filtered = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='same'), axis=y_axis, arr=image)
    filtered = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='same'), axis=x_axis, arr=filtered)
    return filtered


def m(delta):
    return gaussian_convolution(delta, sigma=1, radius=2, y_axis=2, x_axis=3)


def r(m, k=0.06):
    det = m[0,0] * m[1,1] - m[0,1] * m[1,0]
    tr = m[0,0] + m[1,1]
    r = det - k * tr * tr
    return r


if __name__ == "__main__":
    import cv2
    img = cv2.imread(r"C:\Users\Lab\Documents\sakuma\class\computervision\cv_finalreport\target.png")
    im_gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY)

    dst = cv2.cornerHarris(im_gray, 2, 3, 0.04)
    img[dst>0.01*dst.max()]=[0,0,255]
    img[dst<-1000]=[0,255,0]
    cv2.imwrite("dst.png", img)
    print(dst.shape)

    im_gray = im_gray.reshape(*im_gray.shape, 1)
    cv2.imwrite("test_g.png", im_gray)

    delt = delta(im_gray)
    m_mat = m(delt)
    cv2.imwrite("test_m_yy.png", m_mat[0,0])
    cv2.imwrite("test_m_yx.png", m_mat[1,0])
    cv2.imwrite("test_m_xx.png", m_mat[1,1])
    r_mat = r(m_mat)
    print(np.max(r_mat), np.mean(r_mat))
    cv2.imwrite("test_edge.png", np.where(r_mat < 0, 255, 0 ))
    cv2.imwrite("test_corn.png", np.where(r_mat > 323060, 255, 0 ))