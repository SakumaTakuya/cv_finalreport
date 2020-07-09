from itertools import chain

from kivy.graphics import Line, Color, Point
from kivy.properties import BooleanProperty
from kivy.uix.image import Image


from logics.operation import cross, diff

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