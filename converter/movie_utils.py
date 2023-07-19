#!/usr/bin/env python3

import numpy as np
import os
from typing import List, Optional
from converter.movie_info import MovieInfo


def rgb2gray(rgb: "np.ndarray") -> "np.ndarray":

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def get_movies(path: str,
               target: str,
               endings=[".mkv", ".ts"]) -> "np.ndarray":
    movies: List["MovieInfo"] = []
    return np.asarray(rekur(path, movies, target, tuple(endings)))


def rekur(path: str, movies, target: str, endings) -> List["MovieInfo"]:
    for name in os.listdir(path):
        fullPath = os.path.join(path, name)
        if os.path.isdir(fullPath):
            movies = rekur(fullPath, movies, target, endings)
        else:
            if fullPath.endswith(endings) and fullPath.find("_temp") == -1:
                movies.append(MovieInfo(filename=fullPath, target=target))
    return movies
