# imageは全てtextureに合わせた形式で処理

import asyncio
import cv2
import numpy as np
from kivy.app import App
from kivy.clock import mainthread
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty, StringProperty, ListProperty, AliasProperty
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import ScreenManager, Screen
import os
import time
import threading

# from logics.clip import clip_image
from logics.operation import cross, diff
from logics.warp import warp_image_liner, replace_image
from utils.format import cv2tex_format, tex2cv_format
from utils.kivyevent import sleep, popup_task, forget
from utils.mixin import SelectMixin, ImageSelectMixin
from widgets.loaddialog import LoadDialog
from widgets.image import RectDrawImage


class SelectReferenceScreen(ImageSelectMixin, Screen):
    points = ListProperty([])
    next_state = StringProperty("")

    def add_pixels(self, widget, *uv):
        self.points.append(uv)

    def remove_pixels(self):
        self.points.pop(-1)

    def go_next(self):
        self.manager.current = self.next_state

    def show_load(self, load=None, filters=["*.jpg", "*.png"]):
        super().show_load(load, filters)
        if not hasattr(self, "next_button"):
            self.next_button = self.ids.next_button
        self.next_button.disabled = True

    def activate_next(self):
        self.next_button.disabled = False


class SelectDestinationScreen(ImageSelectMixin, Screen):
    texture = ObjectProperty(None)
    next_state = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.texture = Texture.create(size=(1, 1))

    def show_load(self, load=None, filters=["*.jpg", "*.png"]):
        super().show_load(load, filters)
        if not hasattr(self, "next_button"):
            self.next_button = self.ids.next_button
        self.next_button.disabled = False

    def go_next(self):
        self.manager.current = self.next_state


class SelectTargetScreen(SelectMixin, Screen):
    video_source = StringProperty("")

    def set_video_source(self, filename):
        self.video_source = filename[0]

    def show_load(self, load, filters=["*.mkv", "*.ogv", "*.avi", "*.mov", "*.flv"]):
        if load is None:
            load = self.set_video_source
        super().show_load(load, filters)


class TestWidget(Widget):
    min_match_count = 10
    flann_index_kdtree = 0

    def __init__(self):
        super().__init__()

    def set_reference(self, reference, points):
        self.reference = reference
        self.points = points

    def set_destination(self, dest):
        async def task():
            self.destination = dest
            h, w, *_ = dest.shape
            self.reference = await popup_task(
                    "Calculationg...", 
                    warp_image_liner,
                    self.reference, 
                    *self.points[0],
                    *self.points[1],
                    *self.points[2],
                    *self.points[3],
                    h, w)
            sleep(0.333)
        forget(task())

    def set_target(self, target):
        async def task():
            self.target = target
            self.reference = await popup_task(
                "Calculationg...",
                self.execute_image)
        forget(task())

    def execute_image(self):
        sift = cv2.xfeatures2d.SURF_create()
        ref_kp, ref_des = sift.detectAndCompute(self.reference, None)
        frm_kp, frm_des = sift.detectAndCompute(self.target, None)

        index_params = dict(algorithm=self.flann_index_kdtree, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(frm_des, ref_des, k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good) > self.min_match_count:
            src_pts = np.float32([ frm_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ ref_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            # frameからreferenceの変換を取得する
            H, *_ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            ret = replace_image(self.reference, self.target, H)
            cv2.imwrite("result.png", ret)

    def set_video_source(self, source):
        pass


    def execute(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap:
            return

        
        sift = cv2.SIFT()
        ref_kp, ref_des = sift.detectAndCompute(self.reference, None)

        while True:
            ret, frame = cap.read()
            if ret:
                content.value += 1
                frm_kp, frm_des = sift.detectAndCompute(frame, None)

                index_params = dict(algorithm=self.flann_index_kdtree, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(frm_des, ref_des, k=2)

                good = []
                for m,n in matches:
                    if m.distance < 0.7*n.distance:
                        good.append(m)

                if len(good) > self.min_match_count:
                    src_pts = np.float32([ frm_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                    dst_pts = np.float32([ ref_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                    # frameからreferenceの変換を取得する
                    H, *_ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


            else:
                break

        cap.release()


class MatchMoveApp(App):
    def __init__(self):
        super().__init__()
        self.title = "Match Move"

    def build(self):
        return TestWidget()


if __name__ == '__main__':
    MatchMoveApp().run()
