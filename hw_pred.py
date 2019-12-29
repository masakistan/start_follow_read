import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from os import listdir, walk
from os.path import isfile, join

from hw import hw_dataset
from hw import cnn_lstm
from hw.hw_dataset import HwDataset

from utils.dataset_wrapper import DatasetWrapper
from utils import safe_load

from pathlib import Path

import numpy as np
import cv2
import sys
import json
import os
from utils import string_utils, error_rates
import time
import random
import yaml

from utils.dataset_parse import load_file_list
from utils import lm_decoder

def log_softmax(hw):
    line_data = Variable(torch.from_numpy(hw), requires_grad=False)
    softmax_out = torch.nn.functional.log_softmax(line_data, -1).data.numpy()
    return hw

with open(sys.argv[1]) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

img_dir = sys.argv[2]

hw_network_config = config['network']['hw']
pretrain_config = config['pretraining']
img_height = config['network']['hw']['input_height']

char_set_path = hw_network_config['char_set_path']

with open(char_set_path) as f:
    char_set = json.load(f)

idx_to_char = {}
for k,v in char_set['idx_to_char'].items():
    idx_to_char[int(k)] = v

hw = cnn_lstm.create_model(hw_network_config)
hw_path = os.path.join(config['training']['snapshot']['best_validation'], "hw.pt")
hw_state = safe_load.torch_state(hw_path)
hw.load_state_dict(hw_state)
#hw.cuda()

#dtype = torch.cuda.FloatTensor
dtype = torch.FloatTensor

lowest_loss = np.inf

hw.eval()

#img_paths = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
img_paths = list(Path(img_dir).rglob("*.jpg"))
#print(img_paths)

all_preds = []
for img_idx, img_path in enumerate(img_paths):
    img_path = str(img_path)
    img = cv2.imread(img_path)
    img_orig = img
    h, w = img.shape[:2]
    if img is None:
        print("image {} is empty".format(img_path))
        continue

    if img.shape[0] != img_height:
        #if img.shape[0] < img_height:
        #    print "WARNING: upsampling image to fit size"
        percent = float(img_height) / img.shape[0]
        img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)

    img = np.array([img]).transpose([0,3,1,2]).astype(np.float32)
    img = img / 128.0 - 1.0
    #print img
    img = torch.from_numpy(img)
    #img = img.cuda()
    try:
        preds = hw(img).cpu()
    except Exception as e:
        print("exception: {} on image {}".format(e, img_path))
        continue

    output_batch = preds.permute(1,0,2)
    out = output_batch.data.cpu().numpy()

    logits = out[0]
    windows = logits.shape[0]
    window_width = w // windows

    #print(logits.shape)
    vals = np.argmax(logits, axis=1)
    comma_idxs = np.where(vals == 13)[0]
    if len(comma_idxs) > 0:
        #print('comma idx', comma_idxs[0])
        split_idx = comma_idxs[0]
        split_perc = split_idx / windows
        split_px = int(w * split_perc) + window_width
        #print(img_orig.shape)
        surname = img_orig[:, :split_px, :]
        out_path = "test_{}.jpg".format(img_idx)
        #print(out_path)
        cv2.imwrite(out_path, surname)
    pred, raw_pred = string_utils.naive_decode(logits)
    #print(pred)

    pred_str = string_utils.label2str_single(pred, idx_to_char, False)
    dis_path = img_path[img_path.rfind('/') + 1:]
    print(dis_path + '\t' + pred_str)

