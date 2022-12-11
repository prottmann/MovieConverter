#!/usr/bin/env python3

import numpy as np
import os
import json
import pickle
from dataclasses import dataclass, field

from converter.sound_mapper import map_sound


@dataclass(init=True)
class MovieOpts(object):
    """docstring for MovieOpts"""
    quality: int = 20
    preset: str = "slow"
    codec: str = "x264"
    delete_files: bool = False
    detect_logo: bool = False
    shutdown: bool = False
    replace_stereo: bool = False
    endings: list[str] = field(default_factory=list)

    def __init__(self, read=False):
        if read:
            self.readYaml()

        self.endings = [".mkv", ".ts"]

    def readYaml(self):
        import yaml
        with open("converter/config.yaml") as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            p = yaml.load(file, Loader=yaml.FullLoader)
            for key, value in p.items():
                setattr(self, key, value)
                # self.[key] = p[key]


class MovieInfo(object):
    """docstring for MovieInfo"""
    name: str
    targetDir: str

    def __init__(self, filename: str, target: str):
        folder, name, ending = fileparts(filename)
        self.targetDir = target

        temp = os.path.join(folder, (name + "_temp.mkv"))
        cropFile = os.path.join(folder, (name + "_crop.p"))
        jsonFile = os.path.join(folder, (name + ".json"))

        self.filename = filename
        self.temp_vid_name = temp
        self.cropFile = cropFile
        self.name = name
        self.ending = ending
        self.overwrite = "-n"

        self.streamJson = self.loadJson(jsonFile)
        if "tags" in self.streamJson["format"]:
            if "title" in self.streamJson["format"]["tags"]:
                self.name = self.streamJson["format"]["tags"]["title"].replace(
                    ":", "")

    def getOutputName(self):
        special = ""
        i = 0
        outName = os.path.join(self.targetDir, (self.name + special + ".mkv"))
        while os.path.isfile(outName):
            i += 1
            special = f"_{i}"
            outName = os.path.join(self.targetDir,
                                   (self.name + special + ".mkv"))
        return outName

    def loadJson(self, json_filename: str):
        if not os.path.isfile(json_filename):
            logLevel = "-loglevel 8"
            os.system(f'ffprobe {logLevel} -i "{self.filename}" ' +
                      '-show_streams -show_format' +
                      f' -print_format json > "{json_filename}"')
        with open(json_filename, "r") as f:
            loaded_json = json.load(f)
        os.remove(json_filename)
        return loaded_json

    def getCropString(self):
        """ TODO Add deinterlace command in case of interlaced DVD
    """
        print("Loading Cropfile.")
        self.crop = pickle.load(open(self.cropFile, "rb"))
        print(self.crop)
        logo_filter = ""
        """
        if self.crop.logo_box is not None:
            bonus = 2
            self.crop.logo_box = self.crop.logo_box + 2 * np.array(
                [-1, -1, 2, 2])
            logo_filter = f"delogo=x={logo_box[0]}:y={logo_box[1]}:" +
                f"w={logo_box[2]}:h={logo_box[3]},"
        """
        interlace = ""
        for stream in self.streamJson["streams"]:
            print(stream)
            if "codec_type" in stream and "video" in stream["codec_type"]:
                if "field_order" in stream and \
                    stream["field_order"] in ("tt","bt","tb"):
                    interlace = "yadif,"
                break
        crop_filter = f'-filter:v:0 "{interlace}{logo_filter}crop=' +\
            f'{self.crop.hor_size}:{self.crop.vert_size}:' +\
            f'{self.crop.hor_offset}:{self.crop.vert_offset}"'

        return crop_filter

    def getColorString(self):
        # bt601 = "-colorspace 1 -color_trc 1 -color_primaries 1"
        bt709 = "-colorspace 1 -color_trc 1 -color_primaries 1"
        for stream in self.streamJson["streams"]:
            if "video" == stream["codec_type"]:
                if "color_primaries" in stream:
                    if stream["color_primaries"] == "bt709":
                        return bt709

                elif stream["height"] > 700 and stream["width"] == 1920:
                    return bt709
                break
        # Source is Blu Ray
        return ""

    def getCodec(self, params):
        if params.replace_stereo:
            return "copy"
        elif params.codec == "x264":
            return 'libx264 -tune film -x264-params "aq-mode=3:keyint=240"' +\
                f' -crf {params.quality} -preset {params.preset} ' +\
                '-pix_fmt yuv420p'
        elif params.codec == "x265":
            return 'libx265 -x265-params "no-sao=1:aq-mode=3:ctu=32" ' +\
                f'-crf {params.quality} -preset {params.preset} ' +\
                '-pix_fmt yuv420p10le'
        elif params.codec == "svt-av1":
            return f'libsvtav1 -svtav1-params "preset={params.preset}:' +\
                f'crf={params.quality}:keyint=240:film-grain=6:tune=0" ' +\
                '-pix_fmt yuv420p10le'
        raise NotImplementedError

    def process(self, params):
        print(self)
        crop_filter = "" if params.replace_stereo else self.getCropString()
        color_string = self.getColorString()
        codec = self.getCodec(params)

        # Create Map string
        map_str, move = map_sound(self, params.replace_stereo)

        # print(map_str)
        logLevel = "-loglevel error -stats -hide_banner"
        # allgOptions = f"-crf {params['quality']}
        # -preset {params['preset']} {map_str}"

        command = f'ffmpeg {logLevel} {self.overwrite} ' + \
            f'-i "{self.filename}" {map_str} -c:v {codec} ' + \
            f'{color_string} {crop_filter} "{self.getOutputName()}"'

        print(command)
        os.system(command)

        if params.delete_files:
            if os.path.isfile(self.cropFile):
                os.remove(self.cropFile)
            if os.path.isfile(self.filename):
                os.remove(self.filename)

    def __str__(self):
        string = "Name: {0}, Output: {1}".format(self.name,
                                                 self.getOutputName())
        return string


def fileparts(filename: str):
    folder, name = os.path.split(filename)
    name, ending = os.path.splitext(name)
    return folder, name, ending


def rgb2gray(rgb):

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def get_movies(path: str, target: str = None, endings=[".mkv", ".ts"]):
    movies = []
    return np.asarray(rekur(path, movies, target, tuple(endings)))


def rekur(path, movies, target, endings):
    for name in os.listdir(path):
        fullPath = os.path.join(path, name)
        if os.path.isdir(fullPath):
            movies = rekur(fullPath, movies, target, endings)
        else:
            if fullPath.endswith(endings) and fullPath.find("_temp") == -1:
                movies.append(MovieInfo(filename=fullPath, target=target))
    return movies
