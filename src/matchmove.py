import cv2
import numpy as np
from itertools import chain
from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty, StringProperty, ListProperty, BooleanProperty, AliasProperty
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.image import Image
from kivy.graphics import Line, Color, Point
from kivy.uix.screenmanager import ScreenManager, Screen
import os

from widgets.loaddialog import LoadDialog

def cv2tex_format(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img, 0)
    return img

def tex2cv_format(img):
    img = cv2.flip(img, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def diff(x1, y1, x2, y2):
    return x1-x2, y1-y2

def cross(x1, y1, x2, y2):
    return x1*y2-y1*x2

def is_poly(points):    
    vec10 = diff(*points[0],*points[1])
    vec13 = diff(*points[3], *points[1])
    vec12 = diff(*points[2], *points[1])
    return cross(*vec10, *vec13) * cross(*vec13, *vec12) > 0 


class RectDrawImage(Image):
    can_draw = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_event_type('on_add_point')
        self.register_event_type('on_complete')
        self.graphics = []
    
    def draw_rect(self, points):
        if is_poly(points): 
            points.append(points[0])
            poses = list(chain.from_iterable(points))
            Color(0,1,1)
            self.graphics.append(Line(points=poses, width=2))
            self.dispatch("on_complete")
        else:
            self.undo()

    def on_touch_down(self, touch, after=False):
        if not self.can_draw:
            return

        touch_x, touch_y = self.to_widget(*touch.pos, relative=True)
        tex_width, tex_height = self.norm_image_size
        tex_x = (touch_x - (self.width - tex_width) / 2) / tex_width
        tex_y = (touch_y - (self.height - tex_height) / 2) / tex_height
        if  tex_x < 0 or tex_x > 1 or \
            tex_y < 0 or tex_y > 1:
            return

        with self.canvas:
            if len(self.graphics) < 4:
                Color(0,1,1)
                self.graphics.append(Point(points=touch.pos, pointsize=3))
                self.dispatch(
                    "on_add_point", 
                    int(tex_x * self.texture_size[0]), 
                    int(tex_y * self.texture_size[1]))
    
            if len(self.graphics) == 4:
                self.draw_rect([(p.points[0], p.points[1]) for p in self.graphics])

    def on_add_point(self, tex_x, tex_y):
        pass

    def on_complete(self):
        pass

    def undo(self):
        point = self.graphics.pop(-1)
        self.canvas.remove(point)

    def clear(self):
        size = len(self.graphics)
        for i in range(size):
            self.undo()

class SelectMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self, load, filters=["*"]):
        content = LoadDialog(
                    load=load, 
                    cancel=self.dismiss_popup,
                    filters=filters)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(.8, .8))
        self._popup.open()

class ImageSelectMixin(SelectMixin):
    texture = ObjectProperty(Texture.create(size=(1, 1)))
    is_loaded = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cv_img = None

    def load_texture(self, filename):
        img = cv2tex_format(cv2.imread(filename[0]))

        self.cv_img = img
        self.texture = Texture.create(size=(img.shape[1], img.shape[0]))
        self.texture.blit_buffer(img.tobytes())
        self.dismiss_popup()

        self.is_loaded = True

    def show_load(self, load=None, filters=["*.jpg", "*.png"]):
        if load is None:
            load = self.load_texture
        
        super().show_load(load, filters)


class SelectTargetScreen(ImageSelectMixin, Screen):
    points = ListProperty([])
    next_state = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clip_data = None

    def get_clip(self):
        return self.clip_data
    
    def set_clip(self, value):
        self.clip_data = value

    clip = AliasProperty(get_clip, set_clip)

    def add_pixels(self, widget, *uv):
        self.points.append(uv)

    def calculate_clip(self):
        vec01 = diff(*self.points[1], *self.points[0])
        vec12 = diff(*self.points[2], *self.points[1])
        vec23 = diff(*self.points[3], *self.points[2])
        vec30 = diff(*self.points[0], *self.points[3])

        min_x, min_y = np.min(self.points, axis=0)
        max_x, max_y = np.max(self.points, axis=0)
        siz_y = max_y-min_y+1
        siz_x = max_x-min_x+1

        content = ProgressBar(value=0, max=int(siz_y*siz_x))
        self._popup = Popup(title="Calculating...", content=content,
                            size_hint=(.8, .8))
        self._popup.open()

        self.clip = np.zeros(shape=(siz_y, siz_x, 3), dtype=np.uint8)
        for i in range(min_y, max_y+1):
            for j in range(min_x, max_x+1):
                content.value += 1
                if cross(*vec01, *diff(j, i, *self.points[0])) > 0 and \
                   cross(*vec12, *diff(j, i, *self.points[1])) > 0 and \
                   cross(*vec23, *diff(j, i, *self.points[2])) > 0 and \
                   cross(*vec30, *diff(j, i, *self.points[3])) > 0:
                    self.clip[i-min_y, j-min_x] = self.cv_img[i, j]

        self.dismiss_popup()
        self.manager.current = self.next_state

    def show_load(self, load=None, filters=["*.jpg", "*.png"]):
        super().show_load(load, filters)
        self.next_button = self.ids.next_button
        self.next_button.disabled = True

    def activate_next(self):
        self.next_button.disabled = False


class SelectDestinationScreen(ImageSelectMixin, Screen):
    texture = ObjectProperty(Texture.create(size=(1, 1)))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clip_data = None

    def get_clip(self):
        return self.clip_data
    
    def set_clip(self, value):
        self.clip_data = value

    clip = AliasProperty(get_clip, set_clip)

    def on_enter(self, *args):
        img = cv2tex_format(self.clip)
        self.texture = Texture.create(size=(img.shape[1], img.shape[0]))
        self.texture.blit_buffer(img.tobytes())


class TestWidget(Widget):
    def __init__(self):
        super().__init__()


class MatchMoveApp(App):
    def __init__(self):
        super().__init__()
        self.title = "Match Move"

    def build(self):
        return TestWidget()


if __name__ == '__main__':
    MatchMoveApp().run()