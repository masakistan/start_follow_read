import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from os import listdir
from os.path import isfile, join

from hw import hw_dataset
from hw import cnn_lstm
from hw.hw_dataset import HwDataset

from utils.dataset_wrapper import DatasetWrapper
from utils import safe_load

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
    config = yaml.load(f)

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

img_paths = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]

all_preds = []
for img_path in sorted(img_paths):
    img = cv2.imread(os.path.join(img_dir, img_path))

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
    preds = hw(img).cpu()

    output_batch = preds.permute(1,0,2)
    out = output_batch.data.cpu().numpy()

    logits = out[0]
    pred, raw_pred = string_utils.naive_decode(logits)
    pred_str = string_utils.label2str_single(pred, idx_to_char, False)
    print(img_path + '\t' + pred_str)

