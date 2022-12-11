#!/usr/bin/env python3

import os
import sys

from converter.movie_utils import get_movies
from converter.cropper import Cropper

import matplotlib.pyplot as plt


def execute(file_path, target_path, params):
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
    from converter.movie_utils import MovieOpts
    from gui.file_chooser import FileChooser
    from PyQt6.QtWidgets import QApplication
    params = MovieOpts()
    app = QApplication(sys.argv)
    w = FileChooser(params)
