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
from logics.warp import warp_image_liner
from utils.format import cv2tex_format, tex2cv_format
from utils.kivyevent import sleep, popup_task, forget
from utils.mixin import SelectMixin, ImageSelectMixin
from widgets.loaddialog import LoadDialog
from widgets.image import RectDrawImage


class SelectReferenceScreen(ImageSelectMixin, Screen):
    points = ListProperty([])
    next_state = StringProperty("")
    # clip_data = None

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)

    # def get_clip(self):
    #     return self.clip_data
    
    # def set_clip(self, value):
    #     self.clip_data = value

    # clip = AliasProperty(get_clip, set_clip)

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
            print(self.points, self.reference.shape, self.destination.shape)
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
            cv2.imwrite("to_ref.png", self.reference)

        
        forget(task())

    def set_video_source(self, source):
        self.video_source = source


    def execute(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap:
            return

        content = ProgressBar(
            value=0, 
            max=cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.show_popup(content, "Calculating...")
        
        sift = cv2.SIFT()
        kp1, des1 = sift.detectAndCompute(self.reference, None)

        while True:
            ret, frame = cap.read()
            if ret:
                content.value += 1
                kp2, des2 = sift.detectAndCompute(frame, None)
                index_params = dict(algorithm=self.flann_index_kdtree, trees=5)
                search_params = dict(checks=50)

                flann = cv2.FlannBasedMatcher(index_params, search_params)

                matches = flann.knnMatch(des1,des2,k=2)

                good = []
                for m,n in matches:
                    if m.distance < 0.7*n.distance:
                        good.append(m)

                if len(good) > self.min_match_count:
                    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                    M, *_ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                    h,w = img1.shape
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
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
