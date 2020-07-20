# imageは全てtextureに合わせた形式で処理

import asyncio
import cv2
import numpy as np
from kivy.app import App
from kivy.clock import mainthread
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty, StringProperty, ListProperty, AliasProperty, NumericProperty, BooleanProperty
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import ScreenManager, Screen
import os
import time
import threading

# from logics.clip import clip_image
from .logics.operation import cross, diff
from .logics.matching import match_image, get_homography, match_points, detect_keypoint
from .logics.warp import warp_image_liner, replace_image, warp, warp_only
from .utils.file import get_save_path, get_file_name
from .utils.format import cv2tex_format, tex2cv_format
from .utils.kivyevent import sleep, popup_task, forget
from .utils.mixin import SelectMixin, ImageSelectMixin
from .widgets.loaddialog import LoadDialog
from .widgets.image import RectDrawImage

def save_to(to):
    return get_save_path("data", to)

class SelectWidget(ImageSelectMixin, Widget):
    points = ListProperty([])

    def add_pixels(self, widget, *vu):
        self.points.append(np.array(vu))

    def clear_pixels(self):
        self.points = []

    def save(self):
        if len(self.points) < 4:
            return

        image = tex2cv_format(self.cv_img)
        masked = cv2.flip(mask(image, *self.points), 0)

        file_name = get_file_name(self.image_source)

        np.save(save_to(f"points_{file_name}"), np.array(self.points))
        cv2.imwrite(save_to(f"mask_{file_name}.png"), masked)
        cv2.imwrite(save_to(f"base_{file_name}.png"), image)
        cv2.imwrite(
            save_to(f"synth_{file_name}.png"), 
            np.where((masked > 0)[:,:,None], image, 0))

        self.clear_pixels()


def mask(
    image,
    to_bottom_left, to_top_left, to_top_right, to_bottom_right
    ):
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
        return np.where(
            (crs_bl > 0) & (crs_br > 0) & (crs_tr > 0) & (crs_tl > 0), 
            255, 0).astype(np.uint8)

    return np.fromfunction(create, shape=(h, w))


class SelectUtilApp(App):
    def __init__(self):
        super().__init__()
        self.title = "Select Util"

    def build(self):
        return SelectWidget()

if __name__ == '__main__':
    SelectUtilApp().run()