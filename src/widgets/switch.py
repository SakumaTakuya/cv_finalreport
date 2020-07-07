from kivy.properties import ObjectProperty, StringProperty, ListProperty, NumericProperty
from kivy.graphics import Line, Color, Point
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout


class SwitchLayout(FloatLayout):
    now = NumericProperty(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def switch(self, to):
        self.children[self.now].size_hint_x = 0
        self.children[self.now].size_hint_y = 0

        self.children[to].size_hint_x = 1
        self.children[to].size_hint_y = 1
        for child in self.children:
            print(child.size_hint_x)
    