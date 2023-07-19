#!/usr/bin/env python3

import os
import sys

from converter.movie_utils import get_movies
from converter.cropper import Cropper

import matplotlib.pyplot as plt


def execute(file_path, target_path, params) -> None:
    movies = get_movies(file_path, target_path, params.endings)

    for movie in movies:
        if not os.path.isfile(movie.cropFile):
            Cropper(movie, params)

    for movie in movies:
        movie.process(params)

    if params.shutdown:
        if sys.platform == "win32":
            os.system("shutdown /s /t 60")
        else:
            os.system("sudo shutdown")


if __name__ == "__main__":
    from converter.parameters import Parameters
    from gui.file_chooser import FileChooser
    from PyQt6.QtWidgets import QApplication
    params = Parameters()
    params.detect_logo = True
    app = QApplication(sys.argv)
    #source, target = FileChooser().getPaths()
    source = "/mnt/c/Users/peter/Desktop/test"
    target = "/mnt/c/Users/peter/Desktop"
    #w = FileChooser(params)
    execute(source, target, params)