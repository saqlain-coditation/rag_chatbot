import os
import sys


def add_path(dir_path):
    dir_path = os.path.join(".dependencies", dir_path)
    addition = os.path.join(".", dir_path)
    if hasattr(sys, "_MEIPASS"):
        addition = os.path.join(sys._MEIPASS, dir_path)
    os.environ["PATH"] = addition + os.pathsep + os.environ["PATH"]
