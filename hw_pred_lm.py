import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
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
from spellchecker import SpellChecker
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
for k,v in char_set['idx_to_char'].iteritems():
    idx_to_char[int(k)] = v

lm_params = config['network']['lm']
print "Loading LM"
decoder = lm_decoder.LMDecoder(idx_to_char, lm_params)
print "Done Loading LM"

criterion = CTCLoss()

hw = cnn_lstm.create_model(hw_network_config)
hw_path = os.path.join(config['training']['snapshot']['best_validation'], "hw.pt")
hw_state = safe_load.torch_state(hw_path)
hw.load_state_dict(hw_state)
hw.cuda()

dtype = torch.cuda.FloatTensor

lowest_loss = np.inf

hw.eval()

img_paths = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]

#print "Accumulating stats for LM"
#for npz_path in sorted(npz_paths):
#    out = np.load(npz_path)
#    out = dict(out)
#    for o in out['hw']:
#        o = log_softmax(o)
#        decoder.add_stats(o)
#print "Done accumulating stats for LM"

all_preds = []
for img_path in img_paths:
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
    img = img.cuda()
    preds = hw(img).cpu()
    #all_preds.append(preds)
    o = log_softmax(preds.cpu().detach().numpy())
    decoder.add_stats(o)

print 'done!'

for img_path in img_paths:
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
    img = img.cuda()
    preds = hw(img).cpu()
    output_batch = preds.permute(1,0,2)
    out = output_batch.data.cpu().numpy()

    decoded = decoder.decode(log_softmax(preds.cpu().detach().numpy()))
    print ''.join([x[0] for x in decoded])

    logits = out[0]
    pred, raw_pred = string_utils.naive_decode(logits)
    pred_str = string_utils.label2str_single(pred, idx_to_char, False)
    print pred_str
    print '*' * 20
    #words = pred_str.split()
    #for word in words:
    #    print word, spell.correction(word)
    #print '*' * 20

