import types
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar


def forget(coro):
    def step_coro(*args, **kwargs):
        try:
            coro.send((args, kwargs, ))(step_coro)
        except StopIteration:
            pass
    try:
        coro.send(None)(step_coro)
    except StopIteration:
        pass


async def thread(func, *args, **kwargs):
    from threading import Thread
    return_value = None
    is_finished = False
    def wrapper(*args, **kwargs):
        nonlocal return_value, is_finished
        return_value = func(*args, **kwargs)
        is_finished = True
    Thread(target=wrapper, args=args, kwargs=kwargs).start()
    while not is_finished:
        await sleep(0)

    return return_value


@types.coroutine
def sleep(duration):
    args, kwargs = yield lambda step_coro: Clock.schedule_once(step_coro, duration)
    return args[0]


@types.coroutine
def event(ed, name):
    bind_id = None
    step_coro = None
    def bind(step_coro_):
        nonlocal bind_id, step_coro
        bind_id = ed.fbind(name, callback)
        assert bind_id > 0  # bindingに成功したか確認
        step_coro = step_coro_
    def callback(*args, **kwargs):
        ed.unbind_uid(name, bind_id)
        step_coro(*args, **kwargs)
    return (yield bind)


async def popup_task(title, func, *args, **kwargs):
    content = ProgressBar(value=0.7, max=1)
    popup = Popup(title=title, content=content, size_hint=(.8, .1))
    popup.open()
    result = await thread(func, *args, **kwargs)
    popup.dismiss()
    return result

def popup_task_coroutine(coroutine, title,  on_complete=None):
    content = ProgressBar(value=0, max=1)
    popup = Popup(title=title, content=content, size_hint=(.8, .8))
    popup.open()

    def func_update(*args):
        nonlocal content, popup
        try:
            nex = next(coroutine)
            content.value_normalized = nex
            Clock.schedule_once(func_update, 0)
        except StopIteration:
            content.value = 0.0
            popup.dismiss()
            if on_complete is not None:
                on_complete()

    Clock.schedule_once(func_update, 0)


if __name__ == '__main__':
    from kivy.app import App
    from kivy.factory import Factory
    from kivy.uix.button import Button

    class SampleApp(App):
        def build(self):
            return Button()
        def on_start(self):
            forget(self.some_task())
        async def some_task(self):
            def heavy_task():
                import time
                for i in range(5):
                    time.sleep(1)
                    print(i)
            button = self.root
            button.text = 'start heavy task'
            await event(button, 'on_press')
            button.text = 'running...'
            await thread(heavy_task)
            button.text = 'done'

    SampleApp().run()