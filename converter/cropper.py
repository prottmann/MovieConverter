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

from typing import Any
from dataclasses import dataclass

import time


@dataclass(init=True)
class DataBaseEntry(object):
    """docstring for DataBaseEntry"""
    img_size: Any
    logo_box: np.array = None
    logo_cut: np.array = None
    logo_name: str = None


@dataclass(init=True)
class CropOptions(object):
    """docstring for CropOptions"""
    vert_size: int = None
    hor_size: int = None
    vert_offset: int = None
    hor_size: int = None
    logo_box: Any = None


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

        f = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Initiate ORB detector
        orb = cv2.ORB_create()
        if self.params.detect_logo:
            # find the keypoints and descriptors with ORB
            kp1, des1 = orb.detectAndCompute(f, None)
            logo = cv2.adaptiveThreshold(f, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 201, 10)
        count = 0
        while success:
            success, image = vidcap.read()
            if not success:
                break
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if self.params.detect_logo:
                kp2, des2 = orb.detectAndCompute(frame, None)
                l = cv2.adaptiveThreshold(frame, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 201, 10)
                # create BFMatcher object
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                # Match descriptors.
                matches = bf.match(des1, des2)
                # Sort them in the order of their distance.
                matches = sorted(matches, key=lambda x: x.distance)
                i = 0
                for idx, m in enumerate(matches):
                    if m.distance > 0:
                        i = idx
                        break
                matches = matches[:i]
                # Draw first 10 matches.
                img3 = cv2.drawMatches(
                    f,
                    kp1,
                    frame,
                    kp2,
                    matches,
                    None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                # Extract location of good matches
                points1 = np.zeros((len(matches), 2), dtype=np.float32)
                points2 = np.zeros((len(matches), 2), dtype=np.float32)
                for i, match in enumerate(matches):
                    points1[i, :] = kp1[match.queryIdx].pt
                    points2[i, :] = kp2[match.trainIdx].pt
                print(points1, points2)
                plt.imshow(img3), plt.show()

                logo = logo + l

            f = f + frame
            count += 1
            if count > 500:
                break

        #pdb.set_trace()

        self.f = f / f.max()

        if self.params.detect_logo:
            self.logo = logo / logo.max()
            plt.imshow(self.logo)
            plt.show()
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