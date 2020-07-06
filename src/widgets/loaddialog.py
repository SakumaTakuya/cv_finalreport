from kivy.properties import ObjectProperty, StringProperty, ListProperty
from kivy.uix.floatlayout import FloatLayout


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    filters = ListProperty(["*"])