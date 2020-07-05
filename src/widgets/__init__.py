from glob import glob
import os

from kivy.lang import Builder


for kv in glob(f"{os.path.dirname(__file__)}/layouts/*.kv"):
    Builder.load_file(kv)
