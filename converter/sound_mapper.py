#!/usr/bin/env python3

import numpy as np
import cv2
import os
import pickle
from converter import movie_utils

import pdb


def map_sound(movie, replaceStereo: bool = False, defaultLanguage="deu"):
    sprachen = np.array(["deu", "ger", "eng", "jpn", "fra", "fre", ""])
    if not np.isin(defaultLanguage, sprachen):
        defaultLanguage = sprachen[0]

    command = None
    types = []
    languages = []
    idx = []
    channels = []
    sprache = []

    for stream in movie.streamJson["streams"]:
        codec_type = stream["codec_type"]
        if codec_type == "data":
            continue

        if codec_type == "video":
            if "tags" in stream:
                tags = stream["tags"]

                if "language" in tags:
                    languages.append(tags["language"])
                else:
                    languages.append(defaultLanguage)

            else:
                languages.append(defaultLanguage)

        else:

            if "tags" in stream and "language" in stream["tags"]:
                languages.append(stream["tags"]["language"])
            else:
                languages.append("")
                print("Empty language added.")
                #os.remove(json_filename)
                #return "", True
        types.append(codec_type)

        if len(languages) == 1:
            idx.append(0)
        elif types[-1] == types[-2]:
            idx.append(idx[-1] + 1)
        else:
            idx.append(0)

        if "channels" in stream:
            channels.append(stream["channels"])
        else:
            channels.append(0)

        sprache.append(sprachen == languages[-1])

    sprache = np.asarray(sprache)
    types = np.asarray(types)
    languages = np.asarray(languages)
    idx = np.asarray(idx)
    channels = np.asarray(channels)
    keep_languages = np.sum(sprache, axis=1) == 1
    #print(sprache, types, languages, idx, channels, keep_languages)

    main_sprache = "eng"
    for i in range(sprachen.shape[0] - 1, -1, -1):
        if np.sum(sprache[:, i]) > 0:
            main_sprache = sprachen[i]
    print("Main sprache: " + main_sprache)

    idx_stereo = np.stack(
        (sprache[:, np.argwhere(sprachen == main_sprache)[0, 0]], types
         == "audio", channels == 2),
        axis=0)
    idx_stereo = np.all(idx_stereo, axis=0)

    idx_surr = np.stack(
        (sprache[:, np.argwhere(sprachen == main_sprache)[0, 0]], types
         == "audio", channels > 3),
        axis=0)
    idx_surr = np.all(idx_surr, axis=0)

    addStereo = False
    if (np.sum(idx_stereo) == 0 or replaceStereo) and np.sum(idx_surr) > 0:
        addStereo = True
        print("Adding stereo tracks for " + main_sprache + " language")

    if not addStereo and np.sum(keep_languages) == keep_languages.shape[0]:
        print("No need to add stereo or remove languages.")
    if replaceStereo:
        print("Replacing stereo track.")

    new_idx = []
    idx_surr = np.argmax(idx_surr)
    idx_stereo = np.argmax(idx_stereo) if replaceStereo else None
    map_str = ""
    lastType = ""
    for i in range(len(idx)):
        if i == 0:
            new_idx.append(0)
            lastType = types[i]

        else:
            if not keep_languages[i]:
                continue
            if replaceStereo and idx_stereo == i:
                print(f"{idx[i]} skipped")
                continue

            if types[i] == lastType:
                new_idx.append(new_idx[-1] + 1)

            else:
                new_idx.append(0)
                lastType = types[i]

        if types[i] == "video":
            map_str += " -map 0:v "

        elif types[i] == "audio":
            if addStereo and new_idx[-1] == 0:
                stereo = stereo_strings(movie, idx[idx_surr], 0, main_sprache)
                map_str += stereo  #+ center
                new_idx.append(1)

            map_str += f"-map 0:a:{idx[i]} -c:a:{new_idx[-1]} copy "

        elif types[i] == "subtitle":
            map_str += f"-map 0:s:{idx[i]} -c:s:{new_idx[-1]} copy "
        print("Mapping", languages[i], types[i], "from", idx[i], "to",
              new_idx[-1])

    movie.map_str = map_str
    return map_str, False


def stereo_strings(movie, source, target, language):

    def createFilter(target, center=1, front=1, side=0.707, lfe=0.5):
        filter_string = f'-filter:a:{target} "volume=3dB,pan=stereo'
        if movie.streamJson["streams"][
                source +
                1]["channels"] > 6 or "side" in movie.streamJson["streams"][
                    source + 1]["channel_layout"]:
            ch = "S"
        else:
            ch = "B"

        for s in ["L", "R"]:
            filter_string += f"|F{s}<{center}*FC+{front}*F{s}+{side}*{ch}{s}+{lfe}*LFE"
        filter_string += '"'
        return filter_string

    def createMapString(source, target, filter_string, title="Stereo"):
        s = f"-map 0:a:{source} -c:a:{target} aac -b:a:{target} 256k " +\
            f'{filter_string} -metadata:s:a:{target} title="{title}" ' +\
            f"-metadata:s:a:{target} language={language} "

        print(f"Mapping {language} audio from {source} to {target} --- {title}")
        return s

    filter_string = createFilter(target)
    stereo = createMapString(source, target, filter_string)

    target += 1
    filter_string = createFilter(target, center=1, front=0.5, side=0.3, lfe=0.3)
    #center = createMapString(source, target, filter_string, title="Stereo Speak")
    return stereo  #, center
