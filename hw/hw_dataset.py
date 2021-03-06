import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import cv2
import numpy as np
import math

from . import grid_distortion

from utils import string_utils, safe_load, augmentation

import random
PADDING_CONSTANT = 0

def collate(batch):
    batch = [b for b in batch if b is not None]
    #These all should be the same size or error
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1, [b['line_img'].shape[0] for b in batch]
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    dim0 = batch[0]['line_img'].shape[0]
    dim1 = max([b['line_img'].shape[1] for b in batch])
    dim2 = batch[0]['line_img'].shape[2]

    all_labels = []
    label_lengths = []

    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT, dtype = np.float32)
    for i in xrange(len(batch)):
        b_img = batch[i]['line_img']
        input_batch[i,:,:b_img.shape[1],:] = b_img

        l = batch[i]['gt_label']
        all_labels.append(l)
        label_lengths.append(len(l))

    all_labels = np.concatenate(all_labels)
    label_lengths = np.array(label_lengths)

    line_imgs = input_batch.transpose([0,3,1,2])
    line_imgs = torch.from_numpy(line_imgs)
    labels = torch.from_numpy(all_labels.astype(np.int32))
    label_lengths = torch.from_numpy(label_lengths.astype(np.int32))

    return {
        'hw_path': [b['hw_path'] for b in batch],
        "line_imgs": line_imgs,
        "labels": labels,
        "label_lengths": label_lengths,
        "gt": [b['gt'] for b in batch]
    }


class HwDataset(Dataset):
    def __init__(self, set_list, char_to_idx, augmentation=False, img_height=32, random_subset_size=None):

        self.img_height = img_height

        self.ids = set_list
        self.ids.sort()

        self.detailed_ids = []
        for ids_idx, paths in enumerate(self.ids):
            json_path, img_path = paths
            d = safe_load.json_state(json_path)
            if d is None:
                continue
            for i in xrange(len(d)):

                if 'hw_path' not in d[i]:
                    continue
                self.detailed_ids.append((ids_idx, i))

        if random_subset_size is not None:
            self.detailed_ids = random.sample(self.detailed_ids, min(random_subset_size, len(self.detailed_ids)))
        print(len(self.detailed_ids))

        self.char_to_idx = char_to_idx
        self.augmentation = augmentation
        self.warning=False

    def __len__(self):
        return len(self.detailed_ids)

    def __getitem__(self, idx):
        ids_idx, line_idx = self.detailed_ids[idx]
        gt_json_path, img_path = self.ids[ids_idx]
        gt_json = safe_load.json_state(gt_json_path)
        if gt_json is None:
            print("WARNING: no gt json {}".format(img_path))
            return None

        if 'hw_path' not in gt_json[line_idx]:
            print("WARNING: no hw path {}".format(img_path))
            return None

        hw_path = gt_json[line_idx]['hw_path']

        hw_path = hw_path.split("/")[-1:]
        hw_path = "/".join(hw_path)

        hw_folder = os.path.dirname(gt_json_path)

        img = cv2.imread(os.path.join(hw_folder, hw_path))

        if img is None:
            print("WARNING: image is none {}".format(img_path))
            return None

        if img.shape[0] != self.img_height:
            if img.shape[0] < self.img_height and not self.warning:
                self.warning = True
                #print("WARNING: upsampling image to fit size")
            percent = float(self.img_height) / img.shape[0]
            img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)

        if img is None:
            print("WARNING: image is none after upsampling {}".format(img_path))
            return None

        if self.augmentation:
            img = augmentation.apply_random_color_rotation(img)
            img = augmentation.apply_tensmeyer_brightness(img)
            img = grid_distortion.warp_image(img)

        img = img.astype(np.float32)
        img = img / 128.0 - 1.0

        gt = gt_json[line_idx]['gt']
        if len(gt) == 0:
            print("WARNING: gt is empty string {}".format(join(*img_path)))
            return None
        gt_label = string_utils.str2label_single(gt, self.char_to_idx)


        return {
            'hw_path': hw_path,
            "line_img": img,
            "gt": gt,
            "gt_label": gt_label
        }


class HwDataset2(Dataset):
    def __init__(self, set_list, char_to_idx, augmentation=False, img_height=32, random_subset_size=None):

        self.img_height = img_height

        self.ids = [x[0] for x in set_list]
        self.labels = [x[1] for x in set_list]

        print('total items', len(self.ids))

        self.char_to_idx = char_to_idx
        self.augmentation = augmentation
        self.warning=False

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        #ids_idx, line_idx = self.detailed_ids[idx]
        img_path = self.ids[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)

        if img is None:
            print('image is none!', img_path)
            return None

        if img.shape[0] != self.img_height:
            if img.shape[0] < self.img_height and not self.warning:
                self.warning = True
                print("WARNING: upsampling image to fit size")
            percent = float(self.img_height) / img.shape[0]
            img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)

        if img is None:
            print('image resized to none!', img_path)
            return None

        if self.augmentation:
            img = augmentation.apply_random_color_rotation(img)
            img = augmentation.apply_tensmeyer_brightness(img)
            img = grid_distortion.warp_image(img)

        img = img.astype(np.float32)
        img = img / 128.0 - 1.0

        gt = label
        if len(gt) == 0:
            return None
        gt_label = string_utils.str2label_single(gt, self.char_to_idx)


        return {
            'hw_path': img_path,
            "line_img": img,
            "gt": gt,
            "gt_label": gt_label
        }
