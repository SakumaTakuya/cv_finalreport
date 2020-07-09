import cv2


def cv2tex_format(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img, 0)
    return img

def tex2cv_format(img):
    img = cv2.flip(img, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img