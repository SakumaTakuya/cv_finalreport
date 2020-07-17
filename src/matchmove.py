# imageは全てtextureに合わせた形式で処理

import asyncio
import cv2
import numpy as np
from kivy.app import App
from kivy.clock import mainthread
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty, StringProperty, ListProperty, AliasProperty, NumericProperty
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import ScreenManager, Screen
import os
import time
import threading

# from logics.clip import clip_image
from logics.operation import cross, diff
from logics.matching import match_image, get_homography, match_points, detect_keypoint
from logics.warp import warp_image_liner, replace_image, warp
from utils.file import get_save_path
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
        self.dismiss_popup()

    def show_load(self, load=None, filters=["*.mkv", "*.ogv", "*.avi", "*.mov", "*.flv"]):
        if load is None:
            load = self.set_video_source
        super().show_load(load, filters)

        if not hasattr(self, "next_button"):
            self.next_button = self.ids.next_button
        self.next_button.disabled = False

    def go_next(self):
        self.manager.current = self.next_state

class MatchMoveWidget(Widget):
    min_match_count = 10
    flann_index_kdtree = NumericProperty(0)
    video_width = NumericProperty(1024)
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    algorithms = {
        "AKAZE" : cv2.AKAZE_create(),
        "SIFT" : cv2.xfeatures2d.SIFT_create()
    }

    gamma = 1/1.8
    gamma_cvt = np.uint8(255 * (np.linspace(0, 1, 256) ** gamma))
    def correct(self, img):
        return cv2.LUT(img, self.gamma_cvt)

    def save_to(self, to):
        return get_save_path("result", "matchmove", to)

    def set_reference(self, reference, points):
        self.reference = reference
        self.points = np.array(points)

    def set_destination(self, dest):
        async def task():
            self.destination = self.correct(tex2cv_format(dest))
            h, w, *_ = dest.shape
            self.reference = await popup_task(
                    "Calculating...", 
                    warp,
                    self.reference, 
                    self.points[0],
                    self.points[1],
                    self.points[2],
                    self.points[3],
                    np.array([0, 0]),
                    np.array([h, 0]),
                    np.array([h, w]),
                    np.array([0, w]),
                    h, w)
            self.reference = self.correct(tex2cv_format(self.reference))
            cv2.imwrite(self.save_to("destination.png"), self.destination)
            cv2.imwrite(self.save_to("reference.png"), self.reference)
            await sleep(0.333)
        forget(task())

    def set_target(self, target):
        async def task():
            self.target = target
            cv2.imwrite(self.save_to("target.png"), self.target)
            await popup_task(
                "Calculating...",
                self.execute_image)
        forget(task())

    def execute_image(self):
        src_pts, dst_pts = match_image(
            self.reference, 
            self.target, 
            self.min_match_count, self.flann_index_kdtree)

        if src_pts is None or dst_pts is None:
            return

        # frameからreferenceの変換を取得する
        H = get_homography(src_pts, dst_pts)

        ret = replace_image(self.destination, self.target, H)
        cv2.imwrite(self.save_to("result.png"), ret)
        cv2.imwrite(self.save_to("reference.png"), self.reference)
        cv2.imwrite(self.save_to("target.png"), self.target)
        cv2.imwrite(self.save_to("destination.png"), self.destination)

    def set_video_target(self, source, key):
        async def task():
            self.source = source
            start = time.time()
            await popup_task(
                "Calculating...",
                self.execute_video,
                key)
            print(time.time() - start)
        forget(task())

    def execute_video(self, algorithm):
        cap = cv2.VideoCapture(self.source)
        if not cap:
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        writer = cv2.VideoWriter(
            self.save_to(f"result_{algorithm}.mp4"), 
            self.fmt, fps, (self.video_width, self.video_width * h // w))

        ref_kp, ref_des = detect_keypoint(self.reference, self.algorithms[algorithm])
        img=cv2.drawKeypoints(self.reference, ref_kp)
        

        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("\nend process")
                break
            
            frame = cv2.resize(frame, (self.video_width, self.video_width * h // w))
            frame = self.correct(frame)

            print(f"\rdesctipt frame: {i}\t\t\t\t", end="")
            tar_kp, tar_des = detect_keypoint(frame, self.algorithms[algorithm])
            
            print(f"\rmatch frame: {i}\t\t\t\t", end="")
            src_pts, dst_pts = match_points(
                ref_kp, ref_des, 
                tar_kp, tar_des,
                self.min_match_count,
                self.flann_index_kdtree)

            if src_pts is not None or dst_pts is not None:
                # frameからreferenceの変換を取得する
                H = get_homography(src_pts, dst_pts)

                print(f"\rreplace frame: {i}\t\t\t\t", end="")
                frame = replace_image(self.destination, frame, H).astype(np.uint8)

            writer.write(frame)
            i += 1

        writer.release()
        cap.release()


class MatchMoveApp(App):
    def __init__(self):
        super().__init__()
        self.title = "Match Move"

    def build(self):
        return MatchMoveWidget()


if __name__ == '__main__':
    MatchMoveApp().run()
