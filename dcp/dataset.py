#!/usr/bin/python
# -*- coding:utf8 -*-
import os
import math
import json
import random
from PIL import Image
import torch
from pip._vendor.pep517.compat import FileNotFoundError
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torchvision
import torchvision.transforms as transforms
import dcp.transforms as group_transforms
from time import sleep
import numpy as np


class SegmentSampler(Sampler):
    '''
    for training, randomly select one frame in every segment.
    for validation, evenly(one frame per second) select frames in every segment.
    '''
    def __init__(self, data_source, train, batch_size):
        self.data_source = data_source
        self.train = train
        self.batch_size = batch_size
        self.segment_indexes = [list()
                                for _ in range(self.data_source.segment_count)]
        self.fpses = [None, ]*self.data_source.segment_count

        for i, data_pair in enumerate(self.data_source.data_pairs):
            # *_, metadata = data_pair # python2.7 not support *_
            metadata = data_pair[-1]
            self.segment_indexes[metadata['segment_count']].append(i)
            self.fpses[metadata['segment_count']] = metadata['fps']
        self.segment_indexes = list(
            filter(lambda item: len(item) != 0, self.segment_indexes))
        self.fpses = list(filter(lambda item: item is not None, self.fpses))
        assert len(self.segment_indexes) == len(self.fpses)

        # if in validation mode, we can generate the index off-line
        if not train:
            index = []
            for fps, segment_index in zip(self.fpses, self.segment_indexes):
                interval = int(fps) # fpses is [30,30,30,...], segment_indexes is [[0],[1],[2],...], what is this for loop doing?
                for i in range(0, len(segment_index), interval):
                    if interval//2 + i < len(segment_index):
                        index.append(segment_index[interval//2 + i])
                    else: # this is for getting the last one
                        index.append(
                            segment_index[(len(segment_index)-i)//2+i])
            self.index = index

    def __iter__(self):
        if self.train:
            index = [] # this is wrong, every segment_index is a one-item list
            for segment_index in self.segment_indexes:
                index.append(random.choice(segment_index))
            random.shuffle(index)
            return iter(index)
        else:
            return iter(self.index)

    def __len__(self):
        if self.train:
            return len(self.fpses)
        else:
            return len(self.index)


def _pts2index_impl(pts, fps): # get the position of this frame
    fraction, integer = math.modf(pts)
    return int(fps*integer+fraction*100)


def is_rgb(metadata):
    return metadata['type'] == 0


def is_infrared(metadata):
    return metadata['type'] == 1


def is_high_quality(metadata):
    return metadata['quality'] == 0


def is_daytime(metadata):
    return metadata['lighting'] == 0


def is_night(metadata):
    return metadata['lighting'] == 1


def is_in_classes(metadata, labels):
    return metadata['label'] in labels


def is_not_in_classes(metadata, labels):
    return metadata['label'] not in labels


'''
每一行表示一个video segment的信息
{driver: {
    subset: train/test,
    fps: 该segment的帧率
    lighting: 白天/黑夜
    type: RGB/光流
    annotations: {
        segment: 开始结束时刻，例如16.34-18.24：表示第16秒后的第34帧开始，到第18秒后的第24帧结束
        quality: 视频片段质量高低
        label: 动作类别，目前总8类
        }
    }
}
'''

class BusDeriverDataset(Dataset):

    # by deleting these keys, make model better
    anno2index = {
        '1-a-1': 0,  # Normal Driving
        '1-b-1': 0,  # Normal Driving
        '1-c-1': 0,  # Normal Driving
        '2-a-1': 1,  # Drinking or Eating (2-a-1 denotes Drinking)
        '2-b-1': 1,  # Drinking or Eating (2-b-1 denotes Drinking)
        '3-a-1': 1,  # Drinking or Eating (3-a-1 denotes Eating)
        '3-b-1': 1,  # Drinking or Eating (3-a-1 denotes Eating)
        '4-a-1': 2,  # Playing on the Phone
        '4-b-1': 2,  # Playing on the Phone
        '5-a-1': 3,  # Calling
        '5-b-1': 3,  # Calling
        '6-a-1': 4,  # Looking Sideways
        '6-b-1': 4,  # Looking Sideways
        '7-a-1': 5,  # Fighting (7-a-1 denotes fighting for the steering wheel)
        '8-a-1': 5,  # Fighting (8-a-1 denotes pulling the driver)
        '9-a-1': 6,  # Fixing Hair
        '10-a-1': 7,  # Driving with Single Hand
        # '11-a-1': 8,  # Fatigue Driving
    }

    index2name = {
        0: 'Normal_Driving',
        1: 'Drinking_or_Eating',
        2: 'Playing_on_the_Phone',
        3: 'Calling',
        4: 'Looking_Sideways',
        5: 'Fighting',
        6: 'Fixing_Hair',
        7: 'Driving_with_Single_Hand',
        #         8: 'Fatigue_Driving'
    }

    def __init__(self, root, anno_path, train, filters,
                 transforms=None, target_transforms=None, writer=None, **kwargs):
        self.root = root
        self.anno_path = anno_path
        self.train = train
        self.filters = filters
        self.transforms = transforms
        self.target_transforms = target_transforms

        self.segment_count = 0
        self.data_pairs = []

        self._parse_annotation()

        self.writer = writer
        # Here, just to be sure, we check all the frames whether exist in the disk
        for data_pair in self.data_pairs:
            driver, index, target, metadata = data_pair
            path = os.path.join(self.root, driver, "img_{:05d}.jpg".format(index))
            if not os.path.exists(path):
                raise FileNotFoundError("image {} is not found in the disk.".format(path))

        # we are going to filter some frames here
        for f in filters:
            self.data_pairs = [data_pair for data_pair in self.data_pairs
                               if f(data_pair[-1])]
        self.data_pairs = list(self.data_pairs)

    def _parse_annotation(self):
        with open(self.anno_path, 'r') as f:
            self.annotations = json.load(f)

        drivers = sorted(self.annotations.keys())
        # devide into training set and validation set
        subset = 'training' if self.train else 'testing'
        drivers = [
            driver for driver in drivers if self.annotations[driver]['subset'] == subset]

        for driver in drivers:
            fps = round(self.annotations[driver]['fps'])
            if fps == 0 or fps is None:
                raise ValueError("fps should not be {}".format(fps))
            lighting = self.annotations[driver]['lighting']
            type_ = self.annotations[driver]['type']
            for item in self.annotations[driver]['annotations']:
                start_pts, end_pts = item['segment']
                quality = item['quality']
                # Note that pts=16.34 indicates the 34-th frame after 16-th second
                start_index = _pts2index_impl(start_pts, fps)
                end_index = _pts2index_impl(end_pts, fps)
                # first try
                # some process due to the mistakes in annotation file
                if start_index > end_index:
                    continue # start_index = end_index - (start_index - end_index)+1
                total_ims = len((os.listdir('{}/{}'.format(self.root, driver))))
                if end_index > total_ims:
                    end_index = total_ims -1 # gap = np.abs(end_index - total_ims)
                    if start_index > total_ims:
                        continue # start_index -= gap
                    # end_index -= gap
                # second try
                # if start_index >= end_index or end_index-start_index < self.n_frames:
                #     continue

                assert  start_index <= end_index, 'fps={}, start={}, end={}, this {}/{}-{} wrong'.format(fps, start_index, end_index, driver, start_pts, end_pts)
                label = item['label']
                # we only reserve the label anno. in 'BusDeriverDataset.anno2index.keys'
                # this is also a kind of filter
                if label not in BusDeriverDataset.anno2index.keys():
                    continue
                label = BusDeriverDataset.anno2index[label]
                for index in range(start_index, end_index):
                    self.data_pairs.append(
                        (driver, index+1, label, dict(lighting=lighting, type=type_, fps=fps,
                                                      start_pts=start_pts, end_pts=end_pts,
                                                      start_index=start_index, end_index=end_index,
                                                      quality=quality, label=label, segment_count=self.segment_count)))
                self.segment_count += 1 # how many video segments

    def __getitem__(self, index):
        driver, index, target, metadata = self.data_pairs[index]
        image = Image.open(os.path.join(
            self.root, driver, "img_{:05d}.jpg".format(index)))

        if self.transforms is not None:
            image = self.transforms(image)
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return image, target, metadata

    def __len__(self):
        return len(self.data_pairs)

class BusDeriverDataset3D(Dataset):

    anno2index = {
        '1-a-1': 0,  # Normal Driving
        '1-b-1': 0,  # Normal Driving
        '1-c-1': 0,  # Normal Driving
        '2-a-1': 1,  # Drinking or Eating (2-a-1 denotes Drinking)
        '2-b-1': 1,  # Drinking or Eating (2-b-1 denotes Drinking)
        '3-a-1': 1,  # Drinking or Eating (3-a-1 denotes Eating)
        '3-b-1': 1,  # Drinking or Eating (3-a-1 denotes Eating)
        '4-a-1': 2,  # Playing on the Phone
        '4-b-1': 2,  # Playing on the Phone
        '5-a-1': 3,  # Calling
        '5-b-1': 3,  # Calling
        '6-a-1': 4,  # Looking Sideways
        '6-b-1': 4,  # Looking Sideways
        '7-a-1': 5,  # Fighting (7-a-1 denotes fighting for the steering wheel)
        '8-a-1': 5,  # Fighting (8-a-1 denotes pulling the driver)
        '9-a-1': 6,  # Fixing Hair
        '10-a-1': 7,  # Driving with Single Hand
        # '11-a-1': 8,  # Fatigue Driving
    }

    index2name = {
        0: 'Normal_Driving',
        1: 'Drinking_or_Eating',
        2: 'Playing_on_the_Phone',
        3: 'Calling',
        4: 'Looking_Sideways',
        5: 'Fighting',
        6: 'Fixing_Hair',
        7: 'Driving_with_Single_Hand',
        #         8: 'Fatigue_Driving'
    }

    def __init__(self, root, anno_path, train, filters,
                 transforms=None, target_transforms=None,  n_frames=4, interval=0, writer=None):
        self.root = root
        self.anno_path = anno_path
        self.train = train
        self.filters = filters
        self.transforms = transforms
        self.target_transforms = target_transforms

        self.segment_count = 0
        self.data_pairs = []

        self.n_frames = n_frames
        self.interval = interval
        self.writer = writer
        self._parse_annotation()

        # Here, just to be sure, we check all the frames whether exist in the disk
        for data_pair in self.data_pairs:
            driver, start, end, target, metadata = data_pair
            for index in range(start, end):
                path = os.path.join(self.root, driver, "img_{:05d}.jpg".format(index))
                if not os.path.exists(path):
                    raise FileNotFoundError("image {} is not found in the disk.".format(path))

        # we are going to filter some frames here
        for f in filters:
            self.data_pairs = [data_pair for data_pair in self.data_pairs if f(data_pair[-1])]
        self.data_pairs = list(self.data_pairs)

    def _parse_annotation(self):
        with open(self.anno_path, 'r') as f:
            self.annotations = json.load(f)

        drivers = sorted(self.annotations.keys())
        # devide into training set and validation set
        subset = 'training' if self.train else 'testing'
        drivers = [
            driver for driver in drivers if self.annotations[driver]['subset'] == subset]

        for driver in drivers:
            fps = round(self.annotations[driver]['fps'])
            if fps == 0 or fps is None:
                raise ValueError("fps should not be {}".format(fps))
            lighting = self.annotations[driver]['lighting']
            type_ = self.annotations[driver]['type']
            for item in self.annotations[driver]['annotations']:
                start_pts, end_pts = item['segment']
                quality = item['quality']
                # Note that pts=16.34 indicates the 34-th frame after 16-th second
                start_index = _pts2index_impl(start_pts, fps)
                end_index = _pts2index_impl(end_pts, fps)

                # first try
                # some process due to the mistakes in annotation file
                if start_index >= end_index:
                    start_index = end_index - (start_index - end_index)-1
                total_ims = len((os.listdir('{}/{}'.format(self.root, driver))))
                if end_index > total_ims:
                    gap = end_index - total_ims
                    start_index -= gap
                    end_index -= gap
                # second try
                # if start_index >= end_index or end_index-start_index < self.n_frames:
                #     continue

                assert  start_index < end_index, 'fps={}, start={}, end={}, this {}/{}-{} wrong'.format(fps, start_index, end_index, driver, start_pts, end_pts)


                label = item['label']
                # we only rserver the label anno. in 'BusDeriverDataset.anno2index.keys'
                # this is also a kind of filter
                if label not in BusDeriverDataset3D.anno2index.keys():
                    continue
                label = BusDeriverDataset3D.anno2index[label]
                self.data_pairs.append(
                    (driver, start_index+1, end_index+1, label, dict(lighting=lighting, type=type_, fps=fps,
                                                    start_pts=start_pts, end_pts=end_pts,
                                                    start_index=start_index, end_index=end_index,
                                                    quality=quality, label=label, segment_count=self.segment_count)))

                self.segment_count += 1

    def __getitem__(self, index):
        driver, start_index, end_index, target, metadata = self.data_pairs[index]

        n_length = end_index - start_index
        # assert n_length > 0, 'this video {}/{}-{} has 0 frames'.format(driver, start_index, end_index)
        # if self.n_frames + (self.n_frames-1) * self.interval > n_length: # get n_frames with interval, check if this video segment is long enough
        #     raise ValueError(f"can not sample frames from the video clip, n_length={n_length}, n_frames={self.n_frames}, interval={self.interval}")
        images = []
        if self.n_frames + (self.n_frames - 1) * self.interval >= n_length:
            for i in range(self.n_frames):
                if i < n_length:
                    images.append(Image.open(os.path.join(
                        self.root, driver, "img_{:05d}.jpg".format(start_index+self.interval * i + 1))))
                else:
                    images.append(Image.open(os.path.join(
                        self.root, driver, "img_{:05d}.jpg".format(start_index+int(n_length)-1))))
            assert len(images) == self.n_frames, 'still wrong'
        else:
            if self.train:
                sample_length = n_length - (self.n_frames + (
                            self.n_frames - 1) * self.interval)  # how many left after getting the last frame
                sample_start = random.randint(0, sample_length) + start_index # 之前忘加start_index
                for i in range(self.n_frames):
                    images.append(Image.open(os.path.join(
                    self.root, driver, "img_{:05d}.jpg".format(sample_start+self.interval*i+1))))
            else:
                for i in range(self.n_frames):
                    images.append(Image.open(os.path.join(
                        self.root, driver, "img_{:05d}.jpg".format(start_index + n_length // 2 - self.n_frames//2 + self.interval * i + 1))))

        if self.transforms is not None:
            images = self.transforms(images)
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return images, target, metadata

    def __len__(self):
        return len(self.data_pairs)
