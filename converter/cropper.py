#!/usr/bin/env python3

import numpy as np
import cv2
import os
import pickle
from converter import movie_utils

from skimage import measure
from scipy import ndimage

import pdb

import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.ion()

import time


class DataBaseEntry(object):
    """docstring for DataBaseEntry"""

    def __init__(self, img_size, logo_box=None, logo_cut=None, logo_name=None):
        self.img_size = img_size
        self.logo_box = logo_box
        self.logo_cut = logo_cut
        self.logo_name = logo_name

    def __str__(self):
        string = f"Name: {self.logo_name}, Width: {self.width}, "+\
            f"Height: {self.height}"
        return string


class CropOptions(object):
    """docstring for CropOptions"""

    def __init__(self,
                 vert_size=None,
                 hor_size=None,
                 vert_offset=None,
                 hor_offset=None,
                 logo_box=None):
        self.vert_size = vert_size
        self.hor_size = hor_size
        self.vert_offset = vert_offset
        self.hor_offset = hor_size
        self.logo_box = logo_box

    def set_logo(self, logo_box):
        self.logo_box = logo_box

    def set_crop_opts(self,
                      vert_size=None,
                      hor_size=None,
                      vert_offset=None,
                      hor_offset=None):
        self.vert_size = vert_size
        self.hor_size = hor_size
        self.vert_offset = vert_offset
        self.hor_offset = hor_size

    def __str__(self):
        string = f"W: {self.hor_size}, H: {self.vert_size}, "+\
            f"w_off: {self.hor_offset}, h_off: {self.vert_offset}, "+ \
            f"LogoBox: {self.logo_box}"
        return string


class Cropper(object):
    """docstring for Cropper"""

    def __init__(self, movie=None, params=None):
        self.movie = movie
        self.crop_opts = CropOptions()
        self.params = params
        if not params.replace_stereo:
            self.crop_to_file()
            if self.params.delete_files:
                os.remove(self.movie.temp_vid_name)

    def crop_to_file(self):
        logLevel = '-loglevel 8'
        if os.path.isfile(self.movie.cropFile):
            return
        if not os.path.isfile(self.movie.temp_vid_name):
            command = f"ffmpeg {logLevel} {self.movie.overwrite} -ss 300 " +\
              f"-i \"{self.movie.filename}\" -t 120 -c:v copy -preset fast " +\
              f"-r 1 -c:a copy -c:s copy \"{self.movie.temp_vid_name}\""

            os.system(command)
            #print(command)

        self.get_crop_size()
        pickle.dump(self.crop_opts, open(self.movie.cropFile, "wb"))
        self.movie.crop = self.crop_opts

    def get_crop_size(self):
        vidcap = cv2.VideoCapture(self.movie.temp_vid_name)
        #create_Features(self.movie.temp_vid_name)
        success, image = vidcap.read()
        while not success:
            success, image = vidcap.read()
        f = movie_utils.rgb2gray(image)
        if self.params.detect_logo:
            rect, logo = cv2.adaptiveThreshold(f, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 201, 10)
        count = 0
        while success:
            success, image = vidcap.read()
            if not success:
                break
            frame = movie_utils.rgb2gray(image).astype(np.uint8)

            if self.params.detect_logo:
                l = cv2.adaptiveThreshold(frame, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 201, 10)
                logo = logo + l

            f = f + frame
            count += 1
            if count > 500:
                break

        #pdb.set_trace()

        self.f = f / f.max()

        if self.params.detect_logo:
            self.logo = logo / logo.max()
            crop = self.detect_logo()
        else:
            self.crop = None

        f = self.f > 0.05
        self.detect_vert_hor_crop(f)

    def detect_vert_hor_crop(self, f):
        v = np.absolute(np.diff(np.mean(f, axis=1)))
        plt.plot(v)
        self.crop_opts.vert_size, self.crop_opts.vert_offset = self.detect_crop(
            0.6, v)

        h = np.absolute(np.diff(np.mean(
            f, axis=0))) * f.shape[0] / self.crop_opts.vert_size
        plt.plot(h)
        self.crop_opts.hor_size, self.crop_opts.hor_offset = self.detect_crop(
            0.6, h)
        #self.plotDetectedCrop(f)
        print(self.crop_opts)

    def detect_crop(self, threshold, vector):
        upperLimit = np.where(vector[0:int(vector.size / 2)] > threshold)
        lowerLimit = np.where(vector[int(vector.size / 2):] > threshold)

        if upperLimit[0].size == 0:
            offset = 0
        else:
            offset = upperLimit[0][-1]
            if offset < 4:
                offset = 0

        if lowerLimit[0].size == 0:
            frame_size = vector.size + 1 - offset
        else:
            lowerLimit = int(vector.size / 2) + lowerLimit[0][0]
            frame_size = lowerLimit - offset
            if vector.size + 1 - frame_size < 4:
                frame_size = vector.size + 1 - offset
        if frame_size % 2 == 1:
            frame_size += 1
            if offset % 2 == 1:
                offset -= 1
        return frame_size, offset

    def plotDetectedCrop(self, f):
        plt.imshow(f)
        if self.crop_opts.vert_offset > 0:
            plt.plot([0, f.shape[1] - 1],
                     [self.crop_opts.vert_offset, self.crop_opts.vert_offset],
                     linewidth=3)

        val = self.crop_opts.vert_size + self.crop_opts.vert_offset
        if val < f.shape[1] - 1:
            plt.plot([0, f.shape[1] - 1], [val, val], linewidth=3)

        if self.crop_opts.hor_offset > 0:
            plt.plot([0, f.shape[0] - 1],
                     [self.crop_opts.hor_offset, self.crop_opts.hor_offset],
                     linewidth=3)

        val = self.crop_opts.hor_size + self.crop_opts.hor_offset
        if val < f.shape[0] - 1:
            plt.plot([0, f.shape[0] - 1], [val, val], linewidth=3)
        plt.show()

    def detect_logo(self):
        self.read_database()

    def read_database(self):
        if os.path.isfile("database.p"):
            possible_logos = pickle.load(open("database.p", "rb"))
        else:
            possible_logos = []
            possible_logos.append(self.detect_new_logo())
            pickle.dump(possible_logos, open("database.p", "wb"))

        correlation = []
        for entry in possible_logos:
            if self.logo.shape != entry.img_size:
                correlation.append(0)
                continue
            logo_img = np.greater(self.logo, 0.7)
            part = logo_img[entry.logo_box[0]:entry.logo_box[0] +
                            entry.logo_box[2],
                            entry.logo_box[1]:entry.logo_box[1] +
                            entry.logo_box[3]]
            mask = entry.logo_cut
            correlation.append(np.sum((mask > 0) == (part > 0)) / part.size)
        correlation = np.asarray(correlation)
        max_idx = np.argmax(correlation)
        if correlation[max_idx] > 0.75:
            self.logo_box = possible_logos[max_idx].logo_box
        else:
            possible_logos.append(self.detect_new_logo())
            pickle.dump(possible_logos, open("database.p", "wb"))
        pdb.set_trace()

    def detect_new_logo(self):
        l = np.greater(self.logo, 0.85)
        l[int(l.shape[0] / 2):, :] = 0
        labels = measure.label(l.astype(int))
        for i in range(1, labels.max()):
            objs = ndimage.find_objects(labels == i)
            height = int(objs[0][0].stop - objs[0][0].start)
            width = int(objs[0][1].stop - objs[0][1].start)

            if width * height / labels.size > 0.005 or (labels == i).size < 20:
                labels[labels == i] = 0
                continue
            if height > 0.1 * labels.shape[0] or width > 0.1 * labels.shape[1]:
                labels[labels == i] = 0
                continue

        label = labels > 0
        plt.imshow(label)
        plt.show()
        pdb.set_trace()

        objs = ndimage.find_objects(labels)
        logo_box = np.array([])
        for obj in objs:
            height = int(obj[0].stop - obj[0].start)
            width = int(obj[1].stop - obj[1].start)
            if logo_box.size == 0:
                logo_box = np.array([obj[0].start, obj[1].start, height, width])
            else:
                logo_box = np.stack(
                    (logo_box,
                     np.array([obj[0].start, obj[1].start, height, width])),
                    axis=0)
        self.logo_box = np.array([
            logo_box[:, 0].min(), logo_box[:, 1].min(),
            (logo_box[:, 0] + logo_box[:, 2]).max() - logo_box[:, 0].min(),
            (logo_box[:, 1] + logo_box[:, 3]).max() - logo_box[:, 1].min()
        ])
        fig, ax = plt.subplots(1)
        plt.imshow(labels)
        rect = patches.Rectangle((self.logo_box[1], self.logo_box[0]),
                                 height=self.logo_box[2],
                                 width=self.logo_box[3],
                                 linewidth=1,
                                 edgecolor='r')
        ax.add_patch(rect)
        plt.show()
        patch = labels[self.logo_box[0]:self.logo_box[0] + self.logo_box[2],
                       self.logo_box[1]:self.logo_box[1] + self.logo_box[3]]
        plt.imshow(patch)
        plt.show()
        logo_name = input("Enter logo name for new database entry: ")
        return DataBaseEntry(img_size=labels.shape,
                             logo_box=self.logo_box,
                             logo_cut=patch,
                             logo_name=logo_name)