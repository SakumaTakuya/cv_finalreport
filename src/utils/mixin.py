import cv2
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty, StringProperty, ListProperty, BooleanProperty, AliasProperty
from kivy.uix.popup import Popup

from utils.format import cv2tex_format
from widgets.loaddialog import LoadDialog


class SelectMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def show_popup(self, content, title="Load file"):
        self._popup = Popup(title=title, content=content,
                            size_hint=(.8, .8))
        self._popup.open()

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self, load, filters=["*"]):
        content = LoadDialog(
                    load=load, 
                    cancel=self.dismiss_popup,
                    filters=filters)
        self.show_popup(content)


class ImageSelectMixin(SelectMixin):
    texture = ObjectProperty(None)
    is_loaded = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cv_img = None
        self.texture = Texture.create(size=(1, 1))

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