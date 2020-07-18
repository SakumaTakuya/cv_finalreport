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
from logics.operation import cross, diff
from logics.matching import match_image, get_homography, match_points, detect_keypoint
from logics.warp import warp_image_liner, replace_image, warp, warp_only
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

    def set_points_auto(self):
        h, w, *_ = self.cv_img.shape
        self.points = [
            [0, 0], [h, 0], [h, w], [0, w]
        ]

    def show_load(self, load=None, filters=["*.jpg", "*.png"]):
        super().show_load(load, filters)
        if not hasattr(self, "next_button"):
            self.next_button = self.ids.next_button
            self.pass_button = self.ids.pass_button
        self.next_button.disabled = True
        self.pass_button.disabled = False

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
    min_match_count = NumericProperty(10)
    flann_index_kdtree = NumericProperty(0)
    video_width = NumericProperty(1024)
    is_optical = BooleanProperty(False)

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
            h = np.sqrt(np.sum((self.points[1] - self.points[0])**2)).astype(np.int16)
            w = np.sqrt(np.sum((self.points[2] - self.points[1])**2)).astype(np.int16)
            self.destination = cv2.resize(self.correct(tex2cv_format(dest)), (w, h))
            # h, w, *_ = dest.shape
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
            self.reference = tex2cv_format(self.reference)
            cv2.imwrite(self.save_to("destination.png"), self.destination)
            cv2.imwrite(self.save_to("reference.png"), self.reference)
            self.reference = self.correct(self.reference)
            await sleep(0.333)
        forget(task())

    def set_video_target(self, source, key):
        async def task():
            self.source = source
            await popup_task(
                "Calculating...",
                self.execute_video,
                key)
        forget(task())

    def execute_video(self, algorithm, max_speed=1):
        cap = cv2.VideoCapture(self.source)
        if not cap:
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        size_w = self.video_width 
        size_h = self.video_width * h // w

        writer = cv2.VideoWriter(
            self.save_to(f"result_{algorithm}.mp4"), 
            self.fmt, fps, (size_w, size_h))

        ref_kp, ref_des = detect_keypoint(self.reference, self.algorithms[algorithm])
        cv2.imwrite(
            self.save_to(f"keypoints_reference_{algorithm}.png"), 
            cv2.drawKeypoints(self.reference, ref_kp, None, flags=4))

        i = 0
        minh = 0
        minw = 0
        maxh = size_h
        maxw = size_w
        start = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                t = time.time()
                print((t - start) / max(i, 1))
                print("\nend process")
                break
            
            frame = cv2.resize(frame, (size_w, size_h))
            frame = self.correct(frame)

            # print(f"\rdesctipt frame: {i}\t\t\t\t", end="")
            tar_kp, tar_des = detect_keypoint(frame[minh:maxh, minw:maxw], self.algorithms[algorithm])

            # print(f"\rmatch frame: {i}\t\t\t\t", end="")
            src_pts, dst_pts, good = match_points(
                ref_kp, ref_des, 
                tar_kp, tar_des,
                self.min_match_count,
                self.flann_index_kdtree)

            if i == 0:
                # print(f"\save frame: {i}\t\t\t\t", end="")
                cv2.imwrite(
                    self.save_to(f"keypoints_frame_{algorithm}.png"), 
                    cv2.drawKeypoints(frame, tar_kp, None, flags=4))
                cv2.imwrite(
                    self.save_to(f"matches_{algorithm}.png"),
                    cv2.drawMatchesKnn(
                        frame, tar_kp, 
                        self.reference, ref_kp, 
                        good, None,
                        matchColor=(0, 255, 0), matchesMask=None,
                        singlePointColor=(255, 0, 0), flags=0))
                start = time.time()

            if src_pts is not None or dst_pts is not None:
                # frameからreferenceの変換を取得する
                H = get_homography(src_pts, dst_pts)
                if self.is_optical:
                    replaced = warp_only(self.destination, frame, H, minh, minw)
                    mask = np.sum(replaced > 0, axis=2, dtype=bool)

                    # print(f"\rreplace frame: {i}\t\t\t\t", end="")
                    frame = np.where(mask[:,:,None], replaced, frame).astype(np.uint8)
                
                    mask_id = np.array(np.where(mask))
                    minh = min(np.min(mask_id[0])-max_speed, 0)
                    minw = min(np.min(mask_id[1])-max_speed, 0)
                    maxh = min(np.max(mask_id[0])+max_speed, size_h)
                    maxw = min(np.max(mask_id[1])+max_speed, size_w)
                else:
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
